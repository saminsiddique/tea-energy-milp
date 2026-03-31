"""Main entry point for H2-HRES optimization.

Reference: Mulumba & Farzaneh (2025)
"Techno-economic analysis of a hydrogen-based hybrid renewable
energy system for off-grid power supply in Kenya's urban area"
International Journal of Hydrogen Energy 178 (2025) 151474

Usage:
    uv run python main.py                    # Run with optimal config
    uv run python main.py --optimize         # Run full optimization
    uv run python main.py --scenarios        # Run all 8 scenarios
    uv run python main.py --validate         # Run validation against paper
"""

import argparse
import time
from pathlib import Path

import numpy as np

from config.parameters import DEFAULT_PARAMS
from simulation.hourly_simulation import HourlySimulator, SimulationConfig
from simulation.scenario_runner import ScenarioRunner
from optimization.epsilon_constraint import EpsilonConstraintOptimizer
from visualization.pareto_plots import ParetoPlotter
from visualization.dispatch_plots import DispatchPlotter
from visualization.sensitivity_plots import SensitivityPlotter


def run_optimal_simulation(year: int = 2023, h2_price: float = 6.6):
    """Run simulation with paper's optimal configuration (Table 5).

    Optimal capacities from Table 5:
    - PV: 41.8 kW
    - Wind: 30.1 kW
    - Biomass: 27.4 kW
    - Fuel Cell: 15.1 kW
    - Electrolyzer: 40.3 kW
    """
    print("=" * 60)
    print("H2-HRES Simulation with Optimal Configuration")
    print("=" * 60)
    print(f"Year: {year}")
    print(f"H2 Price: ${h2_price}/kg")
    print()

    simulator = HourlySimulator()
    result = simulator.run_optimal_configuration(year, h2_price)

    print("Results:")
    print("-" * 40)
    print(f"COE with H2 market:    ${result.coe_with_h2:.3f}/kWh")
    print(f"COE without H2 market: ${result.coe_without_h2:.3f}/kWh")
    print(f"Reliability:           {result.reliability:.3f}")
    print()
    print(f"Annual energy served:  {result.dispatch_result.total_energy_served/1000:.1f} MWh")
    print(f"Annual H2 produced:    {result.dispatch_result.total_h2_produced:.1f} kg")
    print(f"Annual H2 sold:        {result.dispatch_result.total_h2_sold:.1f} kg")
    print(f"Annual H2 revenue:     ${result.annual_h2_revenue:.0f}")
    print()

    # Capacity factors
    print("Capacity Factors:")
    print(f"  PV:          {result.pv_capacity_factor:.3f}")
    print(f"  Wind:        {result.wind_capacity_factor:.3f}")
    print(f"  Fuel Cell:   {result.fc_utilization:.3f}")
    print(f"  Biomass:     {result.bm_utilization:.3f}")
    print(f"  Electrolyzer:{result.elz_utilization:.3f}")
    print()

    # Validation
    print("Validation against paper (Table 5):")
    print("-" * 40)
    validation = simulator.validate_against_paper(result)
    for key, value in validation.items():
        if key not in ["calculated", "expected"]:
            status = "PASS" if value else "FAIL"
            print(f"  {key}: {status}")

    print()
    print("Calculated vs Expected:")
    for metric in ["coe_with_h2", "coe_without_h2", "reliability"]:
        calc = validation["calculated"][metric]
        exp = validation["expected"][metric]
        print(f"  {metric}: {calc:.3f} (expected: {exp:.3f})")

    return result


def run_all_scenarios(year: int = 2023):
    """Run all 8 scenarios from Table 6."""
    print("=" * 60)
    print("Running All 8 Scenarios (Table 6)")
    print("=" * 60)
    t_start = time.time()

    runner = ScenarioRunner()

    # Run with base H2 price
    print("\nH2 Price: $6.6/kg")
    print("-" * 40)
    results_base = runner.run_optimal_all_scenarios(year, h2_price=6.6)

    print(f"Dry season avg COE:  ${results_base.dry_season_avg_coe:.3f}/kWh")
    print(f"Wet season avg COE:  ${results_base.wet_season_avg_coe:.3f}/kWh")
    print(f"Overall avg COE:     ${results_base.overall_avg_coe:.3f}/kWh")
    print(f"Overall reliability: {results_base.overall_reliability:.3f}")

    # Run with high H2 price
    print("\nH2 Price: $9.9/kg")
    print("-" * 40)
    results_high = runner.run_optimal_all_scenarios(year, h2_price=9.9)

    print(f"Dry season avg COE:  ${results_high.dry_season_avg_coe:.3f}/kWh")
    print(f"Wet season avg COE:  ${results_high.wet_season_avg_coe:.3f}/kWh")
    print(f"Overall avg COE:     ${results_high.overall_avg_coe:.3f}/kWh")
    print(f"Overall reliability: {results_high.overall_reliability:.3f}")

    # Validation
    print("\nValidation against Table 8:")
    print("-" * 40)
    validation_base = runner.validate_seasonal_coe(results_base)
    validation_high = runner.validate_seasonal_coe(results_high)

    for name, val in [("Base ($6.6/kg)", validation_base), ("High ($9.9/kg)", validation_high)]:
        print(f"\n{name}:")
        print(f"  Dry season: {val['calculated']['dry_season_coe']:.3f} (expected: {val['expected']['dry_season_coe']:.3f})")
        print(f"  Wet season: {val['calculated']['wet_season_coe']:.3f} (expected: {val['expected']['wet_season_coe']:.3f})")

    elapsed = time.time() - t_start
    print(f"\nScenarios completed in {elapsed:.1f}s")

    return results_base, results_high


def run_optimization(
    n_pareto_points: int = 10,
    time_limit: int = 600,
    gap: float = 0.005,
    solver_verbose: bool = True,
):
    """Run epsilon-constraint multi-objective optimization."""
    print("=" * 60)
    print("Running Epsilon-Constraint Multi-Objective Optimization")
    print("=" * 60)
    t_start = time.time()

    # Load data
    simulator = HourlySimulator()
    met_data = simulator.load_meteorological_data(2023)
    demand_data = simulator.load_demand_profile(2023)

    # Prepare inputs for optimizer
    demand = demand_data["load_kw"].values

    # Calculate normalized capacity factors
    irradiance = met_data["irradiance"].values
    wind_speed = met_data["wind_speed_50m"].values

    # Normalize to 0-1 range
    irradiance_factor = irradiance / 1000.0  # Normalize by STC
    irradiance_factor = np.clip(irradiance_factor, 0, 1)

    # Wind factor using simplified power curve
    v_ci, v_r, v_co = 3.0, 12.0, 25.0
    wind_factor = np.zeros_like(wind_speed)
    mask1 = (wind_speed >= v_ci) & (wind_speed < v_r)
    wind_factor[mask1] = (wind_speed[mask1]**3 - v_ci**3) / (v_r**3 - v_ci**3)
    mask2 = (wind_speed >= v_r) & (wind_speed <= v_co)
    wind_factor[mask2] = 1.0

    print(f"Demand profile: {len(demand)} hours")
    print(f"Peak demand: {np.max(demand):.1f} kW")
    print(f"Average demand: {np.mean(demand):.1f} kW")
    print(f"Solver config: time_limit={time_limit}s, gap={gap*100:.1f}%, verbose={solver_verbose}")
    print()

    # Load seasonal LHV profile for biomass fuel cost
    from data.fetchers.biomass_data import BiomassDataProvider
    lhv_profile = BiomassDataProvider().get_hourly_lhv(2023)

    # Create optimizer
    optimizer = EpsilonConstraintOptimizer(
        demand_profile=demand,
        irradiance_factor=irradiance_factor,
        wind_factor=wind_factor,
        h2_price=6.6,
        lhv_profile=lhv_profile,
        time_limit_sec=time_limit,
        gap_tolerance=gap,
        solver_verbose=solver_verbose,
    )

    print(f"Generating Pareto front with {n_pareto_points} points...")
    print(f"Note: MILP objective minimizes total cost (H2 revenue decoupled).")
    print(f"      COE = (TC - H2_revenue) / E_served computed post-optimization.")
    print()

    # Generate Pareto front
    pareto_result = optimizer.generate_pareto_front(n_points=n_pareto_points)

    print("Pareto Front Results:")
    print("-" * 60)
    print(f"{'Point':>5} {'COE ($/kWh)':>12} {'Reliability':>12} {'PV (kW)':>10} {'Wind (kW)':>10}")
    print("-" * 60)

    for i, sol in enumerate(pareto_result.solutions):
        knee_marker = " *" if i == pareto_result.knee_point_idx else ""
        print(f"{i+1:>5} {sol.coe:>12.3f} {sol.reliability:>12.3f} {sol.pv_capacity:>10.1f} {sol.wind_capacity:>10.1f}{knee_marker}")

    print()
    print("* = Knee point (optimal trade-off)")
    print()

    knee = pareto_result.knee_point
    print("Knee Point Configuration:")
    print("-" * 40)
    print(f"  PV Capacity:          {knee.pv_capacity:.1f} kW")
    print(f"  Wind Capacity:        {knee.wind_capacity:.1f} kW")
    print(f"  Electrolyzer:         {knee.electrolyzer_capacity:.1f} kW")
    print(f"  Fuel Cell:            {knee.fuel_cell_capacity:.1f} kW")
    print(f"  H2 Storage:           {knee.h2_storage_capacity:.1f} kg")
    print(f"  Biomass:              {knee.biomass_capacity:.1f} kW")
    print(f"  COE:                  ${knee.coe:.3f}/kWh")
    print(f"  Reliability:          {knee.reliability:.3f}")

    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nOptimization completed in {minutes}m {seconds:.1f}s")

    return pareto_result


def run_validation():
    """Run comprehensive validation against paper results."""
    print("=" * 60)
    print("Validation Against Paper Results")
    print("=" * 60)
    t_start = time.time()

    all_passed = True

    # 1. Configuration validation (Table 5)
    print("\n1. Optimal Configuration (Table 5)")
    print("-" * 40)

    expected_config = {
        "PV": 41.8,
        "Wind": 30.1,
        "Biomass": 27.4,
        "Fuel Cell": 15.1,
        "Electrolyzer": 40.3,
    }

    print("Expected optimal capacities (kW):")
    for comp, cap in expected_config.items():
        print(f"  {comp}: {cap}")

    # 2. COE validation
    print("\n2. COE Results Validation")
    print("-" * 40)

    expected_coe = {
        "With H2 @ $6.6/kg": 0.494,
        "Without H2 market": 0.668,
        "With H2 @ $9.9/kg": 0.405,
    }

    simulator = HourlySimulator()

    # Test with base H2 price
    print("  Running base simulation (H2 @ $6.6/kg)...", flush=True)
    result_base = simulator.run_optimal_configuration(2023, 6.6)
    print("  Running simulation without H2 market...", flush=True)
    result_without = simulator.run_optimal_configuration(2023, 0.0)  # No H2 revenue
    print("  Running high H2 price simulation (H2 @ $9.9/kg)...", flush=True)
    result_high = simulator.run_optimal_configuration(2023, 9.9)

    results = {
        "With H2 @ $6.6/kg": result_base.coe_with_h2,
        "Without H2 market": result_base.coe_without_h2,
        "With H2 @ $9.9/kg": result_high.coe_with_h2,
    }

    for scenario, expected in expected_coe.items():
        calculated = results[scenario]
        error_pct = abs(calculated - expected) / expected * 100
        status = "PASS" if error_pct < 15 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {scenario}: ${calculated:.3f}/kWh (expected: ${expected:.3f}, error: {error_pct:.1f}%) [{status}]")

    # 3. Reliability validation
    print("\n3. Reliability Validation")
    print("-" * 40)

    expected_reliability = {
        "With H2 market": 0.961,
        "Without H2 market": 0.978,
    }

    rel_with = result_base.reliability
    rel_without = result_without.reliability

    rel_results = {
        "With H2 market": rel_with,
        "Without H2 market": rel_without,
    }

    for scenario, expected in expected_reliability.items():
        calculated = rel_results[scenario]
        error = abs(calculated - expected)
        status = "PASS" if error < 0.05 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {scenario}: {calculated:.3f} (expected: {expected:.3f}) [{status}]")

    # 4. Seasonal COE validation (Table 8)
    print("\n4. Seasonal COE Validation (Table 8)")
    print("-" * 40)

    runner = ScenarioRunner()
    print("  Running 8 scenarios @ $6.6/kg...", flush=True)
    results_base = runner.run_optimal_all_scenarios(2023, 6.6)
    print("  Running 8 scenarios @ $9.9/kg...", flush=True)
    results_high = runner.run_optimal_all_scenarios(2023, 9.9)

    expected_seasonal = {
        "Dry @ $6.6/kg": 0.452,
        "Wet @ $6.6/kg": 0.511,
        "Dry @ $9.9/kg": 0.398,
        "Wet @ $9.9/kg": 0.442,
    }

    seasonal_results = {
        "Dry @ $6.6/kg": results_base.dry_season_avg_coe,
        "Wet @ $6.6/kg": results_base.wet_season_avg_coe,
        "Dry @ $9.9/kg": results_high.dry_season_avg_coe,
        "Wet @ $9.9/kg": results_high.wet_season_avg_coe,
    }

    for scenario, expected in expected_seasonal.items():
        calculated = seasonal_results[scenario]
        error_pct = abs(calculated - expected) / expected * 100
        status = "PASS" if error_pct < 20 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {scenario}: ${calculated:.3f}/kWh (expected: ${expected:.3f}, error: {error_pct:.1f}%) [{status}]")

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION SUMMARY: ALL TESTS PASSED")
    else:
        print("VALIDATION SUMMARY: SOME TESTS FAILED")
        print("Note: Some deviation is expected due to synthetic data usage")
    print(f"Validation completed in {elapsed:.1f}s")
    print("=" * 60)

    return all_passed


def generate_figures():
    """Generate all figures from the paper."""
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)
    t_start = time.time()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulations
    simulator = HourlySimulator()
    result = simulator.run_optimal_configuration(2023, 6.6)

    runner = ScenarioRunner()
    all_scenarios = runner.run_optimal_all_scenarios(2023, 6.6)

    # Create plotters
    pareto_plotter = ParetoPlotter(output_dir)
    dispatch_plotter = DispatchPlotter(output_dir)
    sensitivity_plotter = SensitivityPlotter(output_dir)

    # Figure 11: Scenario comparison
    print("Generating Figure 11: Scenario comparison...")
    dispatch_plotter.create_figure_11(all_scenarios)

    # Additional dispatch plots
    print("Generating dispatch plots...")
    dispatch_plotter.plot_hourly_dispatch(result, save_path="dispatch_weekly.png")
    dispatch_plotter.plot_h2_dynamics(result, save_path="h2_dynamics.png")
    dispatch_plotter.plot_biomass_operation(result, save_path="biomass_operation.png")
    dispatch_plotter.plot_annual_energy_breakdown(result, save_path="annual_breakdown.png")

    # Sensitivity plots
    print("Generating sensitivity plots...")
    price_results = runner.compare_h2_prices(SimulationConfig(
        pv_capacity_kw=41.8,
        wind_capacity_kw=30.1,
        electrolyzer_capacity_kw=40.3,
        fuel_cell_capacity_kw=15.1,
        h2_storage_capacity_kg=100.0,
        biomass_capacity_kw=27.4,
        year=2023,
    ))
    sensitivity_plotter.plot_h2_price_sensitivity(price_results, save_path="h2_sensitivity.png")
    sensitivity_plotter.plot_seasonal_comparison(all_scenarios, save_path="seasonal_comparison.png")

    elapsed = time.time() - t_start
    print(f"\nFigures saved to: {output_dir} ({elapsed:.1f}s)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="H2-HRES Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py                    Run with optimal configuration
  uv run python main.py --optimize         Run full optimization
  uv run python main.py --scenarios        Run all 8 scenarios
  uv run python main.py --validate         Validate against paper
  uv run python main.py --figures          Generate all figures
        """,
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run epsilon-constraint optimization",
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Run all 8 scenarios",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate against paper results",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate all figures",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Simulation year (default: 2023)",
    )
    parser.add_argument(
        "--h2-price",
        type=float,
        default=6.6,
        help="H2 price in $/kg (default: 6.6)",
    )
    parser.add_argument(
        "--pareto-points",
        type=int,
        default=10,
        help="Number of Pareto points (default: 10)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Max seconds per MILP solve (default: 600)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.005,
        help="Relative optimality gap tolerance (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        default=True,
        help="Show solver output (default: on)",
    )
    parser.add_argument(
        "--solver-quiet",
        action="store_true",
        help="Suppress solver output",
    )

    args = parser.parse_args()

    solver_verbose = args.solver_verbose and not args.solver_quiet

    if args.optimize:
        run_optimization(
            args.pareto_points,
            time_limit=args.time_limit,
            gap=args.gap,
            solver_verbose=solver_verbose,
        )
    elif args.scenarios:
        run_all_scenarios(args.year)
    elif args.validate:
        run_validation()
    elif args.figures:
        generate_figures()
    else:
        run_optimal_simulation(args.year, args.h2_price)


if __name__ == "__main__":
    main()
