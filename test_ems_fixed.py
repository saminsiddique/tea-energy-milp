"""Test Option A (Fig. 3 EMS priority MILP) with Table 6 fixed sizes.

Validation strategy: freeze all capacities to the paper's Table 6 values and
run the dispatch-only MILP. With the EMS priority rules active:
  - Capital cost must match $943,665 exactly (weather-independent).
  - NPC should be close to $1,020,155 (paper). Remaining gap = weather data.
"""

import time
import numpy as np

from data.fetchers.saint_martin_data import SaintMartinDataProvider
from config.nsga_parameters import NSGASystemParams, NSGACapacityBounds, NSGAEconomicParams
from optimization.nsga_milp import NSGAMILPOptimizer


PAPER = {
    "pv_kw": 159, "wind_kw": 100, "dg_kw": 60,
    "batt_kwh": 437, "elz_kw": 139, "gas_storage_m3": 360,
    "npc": 1_020_155, "coe": 0.1724,
    "cow": 1.185, "cog": 3.978,
}
EXPECTED_CAPITAL = (
    159 * 1300 + 100 * 2300 + 60 * 220 + 437 * 158.5
    + 139 * 1200 + 360 * 500 + (159 + 100) * 300
)  # $943,665


def main():
    print("=" * 70)
    print("  Option A: Fig. 3 EMS Priority MILP — Fixed Table 6 Sizes")
    print("=" * 70)

    provider = SaintMartinDataProvider()
    data = provider.load_all_data(2022)
    print(f"\n  Electrical demand : {np.sum(data['elec_demand'])/1000:.0f} MWh/yr")
    print(f"  Water demand      : {np.sum(data['water_demand']):.0f} m3/yr")
    print(f"  Gas demand        : {np.sum(data['gas_demand']):.0f} m3/yr")
    print(f"  PV capacity factor: {np.mean(data['irradiance_factor']):.3f}")
    print(f"  Wind cap factor   : {np.mean(data['wind_factor']):.3f}")

    fixed_bounds = NSGACapacityBounds(
        pv_min=159, pv_max=159,
        wind_min=100, wind_max=100,
        dg_min=60, dg_max=60,
        batt_min=437, batt_max=437,
        elz_min=139, elz_max=139,
        gas_storage_min=360, gas_storage_max=360,
    )
    params = NSGASystemParams(bounds=fixed_bounds)

    optimizer = NSGAMILPOptimizer(
        elec_demand=data["elec_demand"],
        water_demand=data["water_demand"],
        gas_demand=data["gas_demand"],
        irradiance_factor=data["irradiance_factor"],
        wind_factor=data["wind_factor"],
        params=params,
        time_limit_sec=600,
        gap_tolerance=0.01,
        solver_verbose=True,
    )

    print("\n  Solving dispatch MILP with EMS priority constraints...")
    print("  (sizes locked to paper Table 6)")
    t0 = time.time()
    result = optimizer.minimize_npc(lpsp_max=0.05)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  Status      : {result.status}")
    print(f"  Solve time  : {elapsed:.1f} s")
    print("=" * 70)

    if result.status != "Optimal":
        print("  Solver did not find a feasible solution.")
        return

    def pct(v, r):
        return abs(v - r) / abs(r) * 100 if r else 0.0

    print(f"\n  {'Metric':<22} {'MILP':>14} {'Paper':>14} {'Error':>8}")
    print("  " + "-" * 60)
    rows = [
        ("Capital ($)", result.capital, EXPECTED_CAPITAL),
        ("NPC ($)",     result.npc,     PAPER["npc"]),
        ("COE ($/kWh)", result.coe,     PAPER["coe"]),
        ("COW ($/m3)",  result.cow,     PAPER["cow"]),
        ("COG ($/m3)",  result.cog,     PAPER["cog"]),
    ]
    for name, val, ref in rows:
        err = pct(val, ref)
        if abs(val) > 1000:
            print(f"  {name:<22} {val:>14,.0f} {ref:>14,.0f} {err:>7.1f}%")
        else:
            print(f"  {name:<22} {val:>14.4f} {ref:>14.4f} {err:>7.1f}%")

    print(f"\n  Cost Breakdown:")
    print(f"    Capital    : ${result.capital:>12,.0f}")
    print(f"    Replacement: ${result.replacement:>12,.0f}")
    print(f"    O&M        : ${result.om:>12,.0f}")
    print(f"    Fuel       : ${result.fuel_cost:>12,.0f}")
    print(f"    Salvage    :-${result.salvage:>12,.0f}")

    print(f"\n  Annual dispatch:")
    print(f"    Energy served : {result.annual_energy_kwh/1000:.0f} MWh")
    print(f"    DG fuel       : {result.annual_fuel_L:.0f} L")
    print(f"    Unmet         : {result.annual_unmet_kwh:.0f} kWh"
          f" (LPSP={result.lpsp*100:.2f}%)")
    print(f"    Excess dumped : {result.annual_excess_kwh/1000:.0f} MWh")
    print(f"    CH4 produced  : {result.annual_ch4_m3:.0f} m3")

    cap_err = pct(result.capital, EXPECTED_CAPITAL)
    npc_err = pct(result.npc, PAPER["npc"])
    print("\n" + "=" * 70)
    print(f"  Capital match: {'PASS' if cap_err < 1.0 else 'FAIL'}"
          f" (error {cap_err:.2f}%)")
    print(f"  NPC vs paper : {npc_err:.1f}% difference")
    print("=" * 70)


if __name__ == "__main__":
    main()
