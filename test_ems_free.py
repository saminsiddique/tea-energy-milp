"""Test Option A (Fig. 3 EMS priority MILP) with FREE sizing.

Bounds are tightened around paper Table 6 to help Gurobi converge quickly
(the EMS priority formulation uses 35k binaries; loose bounds → weak MIP
relaxation → long solve time).
"""

import time
import numpy as np

from data.fetchers.saint_martin_data import SaintMartinDataProvider
from config.nsga_parameters import NSGASystemParams, NSGACapacityBounds
from optimization.nsga_milp import NSGAMILPOptimizer


PAPER = {
    "pv_kw": 159, "wind_kw": 100, "dg_kw": 60,
    "batt_kwh": 437, "elz_kw": 139, "gas_storage_m3": 360,
    "npc": 1_020_155, "coe": 0.1724,
}
EXPECTED_CAPITAL = (
    159 * 1300 + 100 * 2300 + 60 * 220 + 437 * 158.5
    + 139 * 1200 + 360 * 500 + (159 + 100) * 300
)


def main():
    print("=" * 70)
    print("  Option A: Fig. 3 EMS Priority MILP — Free Sizing")
    print("=" * 70)

    provider = SaintMartinDataProvider()
    data = provider.load_all_data(2022)

    # Bounds tightened around paper Table 6 (±50% around the paper point)
    bounds = NSGACapacityBounds(
        pv_min=80,   pv_max=300,
        wind_min=50, wind_max=200,
        dg_min=0,    dg_max=120,
        batt_min=200,batt_max=800,
        elz_min=50,  elz_max=250,
        gas_storage_min=100, gas_storage_max=600,
    )
    params = NSGASystemParams(bounds=bounds)

    optimizer = NSGAMILPOptimizer(
        elec_demand=data["elec_demand"],
        water_demand=data["water_demand"],
        gas_demand=data["gas_demand"],
        irradiance_factor=data["irradiance_factor"],
        wind_factor=data["wind_factor"],
        params=params,
        time_limit_sec=1200,
        gap_tolerance=0.02,
        solver_verbose=True,
    )

    print("\n  Bounds (PV kW, WT kW, DG kW, Batt kWh, ELZ kW, GS m³):")
    print(f"    PV in [{bounds.pv_min}, {bounds.pv_max}]")
    print(f"    WT in [{bounds.wind_min}, {bounds.wind_max}]")
    print(f"    DG in [{bounds.dg_min}, {bounds.dg_max}]")
    print(f"    Batt in [{bounds.batt_min}, {bounds.batt_max}]")
    print(f"    ELZ in [{bounds.elz_min}, {bounds.elz_max}]")
    print(f"    GS in [{bounds.gas_storage_min}, {bounds.gas_storage_max}]")

    print("\n  Solving free-sizing MILP with EMS priority (LPSP <= 5%)...")
    t0 = time.time()
    result = optimizer.minimize_npc(lpsp_max=0.05)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  Status      : {result.status}")
    print(f"  Solve time  : {elapsed:.1f} s")
    print("=" * 70)

    if result.status != "Optimal":
        print("  No feasible solution found.")
        return

    def pct(v, r):
        return abs(v - r) / abs(r) * 100 if r else 0.0

    print(f"\n  {'Component':<20} {'MILP':>12} {'Paper':>12} {'Diff':>8}")
    print("  " + "-" * 56)
    comps = [
        ("PV (kW)",          result.pv_kw,     PAPER["pv_kw"]),
        ("Wind (kW)",        result.wind_kw,   PAPER["wind_kw"]),
        ("DG (kW)",          result.dg_kw,     PAPER["dg_kw"]),
        ("Battery (kWh)",    result.batt_kwh,  PAPER["batt_kwh"]),
        ("Electrolyzer (kW)",result.elz_kw,    PAPER["elz_kw"]),
        ("Gas Storage (m³)", result.gas_storage_m3, PAPER["gas_storage_m3"]),
        ("NPC ($)",          result.npc,       PAPER["npc"]),
        ("COE ($/kWh)",      result.coe,       PAPER["coe"]),
    ]
    for name, v, r in comps:
        e = pct(v, r)
        if abs(v) > 1000:
            print(f"  {name:<20} {v:>12,.0f} {r:>12,.0f} {e:>7.1f}%")
        else:
            print(f"  {name:<20} {v:>12.2f} {r:>12.2f} {e:>7.1f}%")

    print(f"\n  Cost Breakdown:")
    print(f"    Capital    : ${result.capital:>12,.0f}")
    print(f"    Replacement: ${result.replacement:>12,.0f}")
    print(f"    O&M        : ${result.om:>12,.0f}")
    print(f"    Fuel       : ${result.fuel_cost:>12,.0f}")
    print(f"    Salvage    :-${result.salvage:>12,.0f}")

    print(f"\n  Dispatch: DG fuel={result.annual_fuel_L:.0f} L, "
          f"LPSP={result.lpsp*100:.2f}%, "
          f"Excess={result.annual_excess_kwh/1000:.0f} MWh")


if __name__ == "__main__":
    main()
