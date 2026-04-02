"""Entry point for NSGA-II paper replication using MILP.

Reference: Ahmed et al. (2024) - ECM 299, 117865
"Energy management and sizing of a stand-alone hybrid renewable energy
system for community electricity, fresh water, and cooking gas demands
of a remote island"

Usage:
    uv run python nsga_main.py
    uv run python nsga_main.py --time-limit 900
"""

import argparse
import time

import numpy as np

from data.fetchers.saint_martin_data import SaintMartinDataProvider
from optimization.nsga_milp import NSGAMILPOptimizer
from config.nsga_parameters import DEFAULT_NSGA_PARAMS


# Paper Table 6 targets (PV/WT/DG/Batt configuration)
PAPER_TARGETS = {
    "PV (kW)": 159,
    "Wind (kW)": 100,
    "DG (kW)": 60,
    "Battery (kWh)": 437,
    "Electrolyzer (kW)": 139,
    "Gas Storage (m³)": 360,
    "COE ($/kWh)": 0.1724,
    "NPC ($)": 1_020_155,
    "COW ($/m³)": 1.185,
    "COG ($/m³)": 3.978,
}


def main():
    parser = argparse.ArgumentParser(description="NSGA Paper MILP Optimizer")
    parser.add_argument("--time-limit", type=int, default=600, help="Solver time limit (s)")
    parser.add_argument("--gap", type=float, default=0.01, help="MIP gap tolerance")
    parser.add_argument("--lpsp", type=float, default=0.01, help="Max LPSP (0.01 = 1%%)")
    parser.add_argument("--year", type=int, default=2022, help="Data year")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver output")
    args = parser.parse_args()

    print("=" * 60)
    print("NSGA-II Paper Replication using Epsilon-Constraint MILP")
    print("Ahmed et al. (2024) - ECM 299, 117865")
    print("=" * 60)

    # Load data
    print("\nLoading Saint Martin Island data...")
    provider = SaintMartinDataProvider()
    data = provider.load_all_data(args.year)

    elec = data["elec_demand"]
    water = data["water_demand"]
    gas = data["gas_demand"]
    irr = data["irradiance_factor"]
    wf = data["wind_factor"]

    print(f"  Electrical demand: {np.sum(elec)/1000:.0f} MWh/yr (avg {np.mean(elec):.1f} kW)")
    print(f"  Water demand: {np.sum(water):.0f} m³/yr ({np.mean(water)*24:.1f} m³/day)")
    print(f"  Gas demand: {np.sum(gas):.0f} m³/yr ({np.mean(gas)*24:.1f} m³/day)")
    print(f"  PV capacity factor: {np.mean(irr):.3f}")
    print(f"  Wind capacity factor: {np.mean(wf):.3f}")

    # Create optimizer
    print(f"\nSolver: Gurobi, time_limit={args.time_limit}s, gap={args.gap*100:.1f}%")
    optimizer = NSGAMILPOptimizer(
        elec_demand=elec,
        water_demand=water,
        gas_demand=gas,
        irradiance_factor=irr,
        wind_factor=wf,
        time_limit_sec=args.time_limit,
        gap_tolerance=args.gap,
        solver_verbose=not args.quiet,
    )

    # Run optimization
    print(f"\nMinimizing NPC with LPSP <= {args.lpsp*100:.1f}%...")
    t0 = time.time()
    result = optimizer.minimize_npc(lpsp_max=args.lpsp)
    elapsed = time.time() - t0

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Solve time: {elapsed:.1f}s")

    if result.status != "Optimal":
        print("Optimization failed!")
        return

    print(f"\n{'Metric':<25} {'MILP':>12} {'Paper':>12} {'Error':>8}")
    print("-" * 60)

    comparisons = [
        ("PV (kW)", result.pv_kw, 159),
        ("Wind (kW)", result.wind_kw, 100),
        ("DG (kW)", result.dg_kw, 60),
        ("Battery (kWh)", result.batt_kwh, 437),
        ("Electrolyzer (kW)", result.elz_kw, 139),
        ("Gas Storage (m³)", result.gas_storage_m3, 360),
        ("COE ($/kWh)", result.coe, 0.1724),
        ("NPC ($)", result.npc, 1_020_155),
        ("LPSP", result.lpsp, 0.0),
        ("Reliability", result.reliability, 1.0),
    ]

    for name, val, target in comparisons:
        if target > 0:
            err = abs(val - target) / target * 100
            err_str = f"{err:.1f}%"
        else:
            err_str = "-"
        if isinstance(val, float) and val > 1000:
            print(f"  {name:<23} {val:>12,.0f} {target:>12,.0f} {err_str:>8}")
        else:
            print(f"  {name:<23} {val:>12.2f} {target:>12.2f} {err_str:>8}")

    print(f"\nCost Breakdown:")
    print(f"  Capital:     ${result.capital:,.0f}")
    print(f"  Replacement: ${result.replacement:,.0f}")
    print(f"  O&M:         ${result.om:,.0f}")
    print(f"  Fuel:        ${result.fuel_cost:,.0f}")
    print(f"  Salvage:     -${result.salvage:,.0f}")

    print(f"\nAnnual Performance:")
    print(f"  Energy served: {result.annual_energy_kwh/1000:.0f} MWh")
    print(f"  Fuel consumed: {result.annual_fuel_L:.0f} L/yr")
    print(f"  Gas produced:  {result.annual_ch4_m3:.0f} m³/yr")
    print(f"  Unmet energy:  {result.annual_unmet_kwh:.0f} kWh ({result.lpsp*100:.2f}%)")
    print(f"  Excess energy: {result.annual_excess_kwh/1000:.0f} MWh")
    total_gas_demand = np.sum(gas)
    gas_satisfaction = result.annual_ch4_m3 / total_gas_demand * 100 if total_gas_demand > 0 else 0
    print(f"  Gas satisfaction: {gas_satisfaction:.1f}% of {total_gas_demand:.0f} m3/yr demand")


if __name__ == "__main__":
    main()
