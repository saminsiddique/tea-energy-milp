"""Entry point for paper replication via NSGA-II + rule-based EMS (Option B).

Ahmed et al. (2024) — ECM 299, 117865
"Energy management and sizing of a stand-alone hybrid renewable energy system
for community electricity, fresh water, and cooking gas demands of a remote
island"

Two-stage run:
  1. Fixed Table 6 sizes — verifies cost equations (fast, <1 s after numba JIT).
  2. Full NSGA-II free sizing — 500 pop x 500 gens, paper Table 5 parameters.

Usage:
    uv run python nsga2_main.py                  # both stages
    uv run python nsga2_main.py --fixed-only     # just the cost check
    uv run python nsga2_main.py --gens 100       # quick NSGA-II run
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from config.nsga_parameters import (
    DEFAULT_NSGA_PARAMS,
    NSGACapacityBounds,
    NSGASystemParams,
)
from data.fetchers.saint_martin_data import SaintMartinDataProvider
from optimization.ems_simulator import simulate, SimulationResult
from optimization.nsga2_optimizer import run_nsga2, NSGA2RunSummary


# With the corrected EMS (Fig. 3: RO + gas powered from excess/DG, not base
# load), the optimizer should naturally find reasonable component sizes without
# artificial tight bounds.  Default bounds from NSGACapacityBounds are used.


PAPER = {
    "pv_kw": 159.0,
    "wind_kw": 100.0,
    "dg_kw": 60.0,
    "batt_kwh": 437.0,
    "elz_kw": 139.0,
    "gas_storage_m3": 360.0,
    "npc": 1_020_155.0,
    "coe": 0.1724,
    "cow": 1.185,
    "cog": 3.978,
}


def pct(v: float, r: float) -> float:
    return abs(v - r) / abs(r) * 100.0 if r else 0.0


def print_result(result: SimulationResult, label: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 66)
    rows = [
        ("PV (kW)",          result.pv_kw,          PAPER["pv_kw"]),
        ("Wind (kW)",        result.wind_kw,        PAPER["wind_kw"]),
        ("DG (kW)",          result.dg_kw,          PAPER["dg_kw"]),
        ("Battery (kWh)",    result.batt_kwh,       PAPER["batt_kwh"]),
        ("Electrolyzer (kW)",result.elz_kw,         PAPER["elz_kw"]),
        ("Gas Storage (m3)", result.gas_storage_m3, PAPER["gas_storage_m3"]),
        ("NPC ($)",          result.npc,            PAPER["npc"]),
        ("COE ($/kWh)",      result.coe,            PAPER["coe"]),
        ("COW ($/m3)",       result.cow,            PAPER["cow"]),
        ("COG ($/m3)",       result.cog,            PAPER["cog"]),
    ]
    print(f"  {'Metric':<22} {'Ours':>14} {'Paper':>14} {'Error':>8}")
    for name, val, ref in rows:
        e = pct(val, ref)
        if abs(val) > 1000:
            print(f"  {name:<22} {val:>14,.0f} {ref:>14,.0f} {e:>7.1f}%")
        else:
            print(f"  {name:<22} {val:>14.4f} {ref:>14.4f} {e:>7.1f}%")

    print(f"\n  Cost breakdown:")
    print(f"    Capital     : ${result.capital:>12,.0f}")
    print(f"    Replacement : ${result.replacement:>12,.0f}")
    print(f"    O&M         : ${result.om:>12,.0f}")
    print(f"    Fuel        : ${result.fuel_cost:>12,.0f}")
    print(f"    RO total    : ${result.ro_total:>12,.0f}")
    print(f"    Salvage     :-${result.salvage:>12,.0f}")
    print(f"    -----------------------------")
    print(f"    NPC         : ${result.npc:>12,.0f}")

    print(f"\n  Dispatch:")
    print(f"    Energy served : {result.annual_energy_kwh/1000:.0f} MWh")
    print(f"    DG hours      : {result.annual_dg_hours} h")
    print(f"    DG fuel       : {result.annual_fuel_L:,.0f} L")
    print(f"    Unmet (LPSP)  : {result.annual_unmet_kwh:,.0f} kWh ({result.lpsp*100:.2f}%)")
    print(f"    Excess dumped : {result.annual_excess_kwh/1000:.0f} MWh")
    print(f"    CH4 produced  : {result.annual_ch4_m3_stp:,.0f} m3 STP")
    print(f"    CH4 delivered : {result.annual_ch4_delivered_stp:,.0f} m3 STP")
    print(f"    Renewable fraction: {result.renewable_fraction*100:.1f}%")


def stage1_fixed(data: dict, params: NSGASystemParams) -> SimulationResult:
    """Run simulator with paper Table 6 sizes to validate cost equations."""
    print("\n" + "=" * 70)
    print("  STAGE 1 — Fixed Table 6 sizes (cost model validation)")
    print("=" * 70)

    sizes = {
        "pv_kw": PAPER["pv_kw"],
        "wind_kw": PAPER["wind_kw"],
        "dg_kw": PAPER["dg_kw"],
        "batt_kwh": PAPER["batt_kwh"],
        "elz_kw": PAPER["elz_kw"],
        "gas_storage_m3": PAPER["gas_storage_m3"],
    }

    t0 = time.time()
    result = simulate(sizes, data, params)
    elapsed = time.time() - t0
    print(f"\n  Simulated in {elapsed*1000:.1f} ms")
    print_result(result, "Fixed Table 6 sizing vs paper")
    return result


def export_pareto_csv(population, filepath: str) -> None:
    """Write the full final population to a CSV file."""
    import csv
    fields = ["rank", "feasible", "pv_kw", "wind_kw", "dg_kw",
              "batt_kwh", "elz_kw", "gas_storage_m3",
              "npc", "lpsp_pct", "gas_violation"]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(population):
            w.writerow({
                "rank": i + 1,
                "feasible": r.feasible,
                "pv_kw": f"{r.pv_kw:.2f}",
                "wind_kw": f"{r.wind_kw:.2f}",
                "dg_kw": f"{r.dg_kw:.2f}",
                "batt_kwh": f"{r.batt_kwh:.2f}",
                "elz_kw": f"{r.elz_kw:.2f}",
                "gas_storage_m3": f"{r.gas_storage_m3:.2f}",
                "npc": f"{r.npc:.0f}",
                "lpsp_pct": f"{r.lpsp * 100:.4f}",
                "gas_violation": f"{r.gas_violation:.2f}",
            })


def stage2_nsga2(data: dict, params: NSGASystemParams, pareto_csv: str = "pareto_front.csv") -> NSGA2RunSummary:
    """Run NSGA-II sizing optimization with default (open) bounds."""
    cfg = params.nsga2
    b = params.bounds
    print("\n" + "=" * 70)
    print(f"  STAGE 2 — NSGA-II sizing (pop={cfg.pop_size}, gens={cfg.n_gens})")
    print(f"  SBX p={cfg.p_crossover} eta={cfg.sbx_eta}, "
          f"PM p={cfg.p_mutation} eta={cfg.pm_eta}, LPSP_max={cfg.lpsp_max*100:.1f}%, "
          f"seed={cfg.seed}")
    print(f"  Bounds (default, open):")
    print(f"    PV   in [{b.pv_min},{b.pv_max}]  WT in [{b.wind_min},{b.wind_max}]  "
          f"DG in [{b.dg_min},{b.dg_max}]")
    print(f"    Batt in [{b.batt_min},{b.batt_max}]  ELZ in [{b.elz_min},{b.elz_max}]  "
          f"GS in [{b.gas_storage_min},{b.gas_storage_max}]")
    print("=" * 70)

    t0 = time.time()
    summary = run_nsga2(data=data, params=params, bounds=None, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  NSGA-II finished: {summary.n_evals} evaluations in "
          f"{elapsed/60:.1f} min ({elapsed/summary.n_evals*1000:.2f} ms/eval)")
    print_result(summary.best, "NSGA-II best solution vs paper")

    pop = summary.final_population
    n_total = len(pop)
    n_feas = sum(1 for r in pop if r.feasible)
    npc_vals = [r.npc for r in pop]
    best_lpsp = min(r.lpsp for r in pop if r.feasible) if n_feas else float("nan")
    print(f"\n  Population export:")
    print(f"    Total members  : {n_total}")
    print(f"    Feasible       : {n_feas}  ({n_feas/n_total*100:.0f}%)")
    print(f"    NPC range      : ${min(npc_vals):,.0f} – ${max(npc_vals):,.0f}")
    print(f"    Best LPSP      : {best_lpsp*100:.4f}%")
    export_pareto_csv(pop, pareto_csv)
    print(f"  Saved: {pareto_csv}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-only", action="store_true",
                        help="Skip NSGA-II stage, only run fixed-sizes check")
    parser.add_argument("--gens", type=int, default=None,
                        help="Override Table 5 generation count (quick test)")
    parser.add_argument("--pop", type=int, default=None,
                        help="Override Table 5 population size (quick test)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override NSGA-II seed")
    parser.add_argument("--year", type=int, default=2022,
                        help="NASA POWER data year")
    parser.add_argument("--pareto-csv", default="pareto_front.csv",
                        help="Output CSV path for final population export")
    args = parser.parse_args()

    print("=" * 70)
    print("  Ahmed et al. (2024) paper replication — Option B")
    print("  NSGA-II + rule-based EMS (Fig. 3)")
    print("=" * 70)

    # Load params (allow overrides)
    params = DEFAULT_NSGA_PARAMS
    if args.gens is not None:
        params.nsga2.n_gens = args.gens
    if args.pop is not None:
        params.nsga2.pop_size = args.pop
    if args.seed is not None:
        params.nsga2.seed = args.seed

    # Load data
    print(f"\n  Loading Saint Martin Island data for {args.year}...")
    provider = SaintMartinDataProvider()
    data = provider.load_all_data(args.year)
    print(f"    Electrical demand : {np.sum(data['elec_demand'])/1000:.0f} MWh/yr")
    print(f"    Water demand      : {np.sum(data['water_demand']):.0f} m3/yr")
    print(f"    Gas demand        : {np.sum(data['gas_demand']):.0f} m3/yr")
    print(f"    PV cap factor     : {np.mean(data['irradiance_factor']):.3f}")
    print(f"    Wind cap factor   : {np.mean(data['wind_factor']):.3f}")

    # Stage 1 — fixed Table 6 validation
    fixed_result = stage1_fixed(data, params)

    if args.fixed_only:
        print("\n  (--fixed-only) skipping NSGA-II stage.")
        return

    # Stage 2 — NSGA-II free optimization
    nsga_summary = stage2_nsga2(data, params, pareto_csv=args.pareto_csv)
    nsga_result = nsga_summary.best

    # Final comparison
    print("\n" + "=" * 70)
    print("  FINAL — Fixed vs NSGA-II vs Paper")
    print("=" * 70)
    print(f"  {'Metric':<22} {'Fixed':>14} {'NSGA-II':>14} {'Paper':>14}")
    rows = [
        ("PV (kW)",       fixed_result.pv_kw,       nsga_result.pv_kw,       PAPER["pv_kw"]),
        ("Wind (kW)",     fixed_result.wind_kw,     nsga_result.wind_kw,     PAPER["wind_kw"]),
        ("DG (kW)",       fixed_result.dg_kw,       nsga_result.dg_kw,       PAPER["dg_kw"]),
        ("Battery (kWh)", fixed_result.batt_kwh,    nsga_result.batt_kwh,    PAPER["batt_kwh"]),
        ("ELZ (kW)",      fixed_result.elz_kw,      nsga_result.elz_kw,      PAPER["elz_kw"]),
        ("GS (m3)",       fixed_result.gas_storage_m3, nsga_result.gas_storage_m3, PAPER["gas_storage_m3"]),
        ("NPC ($)",       fixed_result.npc,         nsga_result.npc,         PAPER["npc"]),
        ("COE ($/kWh)",   fixed_result.coe,         nsga_result.coe,         PAPER["coe"]),
    ]
    for name, a, b, r in rows:
        if abs(r) > 1000:
            print(f"  {name:<22} {a:>14,.0f} {b:>14,.0f} {r:>14,.0f}")
        else:
            print(f"  {name:<22} {a:>14.4f} {b:>14.4f} {r:>14.4f}")


if __name__ == "__main__":
    main()
