"""Inspect data at the infeasibility cluster (hours 2993-3000)."""
import numpy as np
from data.fetchers.saint_martin_data import SaintMartinDataProvider
from config.nsga_parameters import DEFAULT_NSGA_PARAMS

provider = SaintMartinDataProvider()
data = provider.load_all_data(2022)

p = DEFAULT_NSGA_PARAMS
pv = 159
wt = 100
ro_per_m3 = p.ro.specific_energy
irr = data["irradiance_factor"]
w = data["wind_factor"]
elec = data["elec_demand"]
water = data["water_demand"]

print(f"{'h':>5} {'elec':>7} {'RO':>6} {'base':>7} {'re_h':>7} {'def':>7} {'head0':>7}")
# Assume a battery state "just before" to see headroom needs
for h in range(2985, 3005):
    re_h = pv * irr[h] + wt * w[h]
    ro = water[h] * ro_per_m3
    base = elec[h] + ro
    defc = max(0.0, base - re_h)
    surp = max(0.0, re_h - base)
    print(f"{h:>5} {elec[h]:>7.1f} {ro:>6.2f} {base:>7.2f} {re_h:>7.2f} "
          f"{defc:>7.2f} {surp:>7.2f}")

# Battery specs
print(f"\nBattery: 437 kWh, SOC_min=20% -> usable 349.6 kWh")
print(f"Max discharge AC kWh/h = 349.6 (if cap_rate unlimited)")
print(f"DG max = 60 kW, so per-hour max supply (batt empty) = 60 + elec_demand max unmet")

# What's the peak deficit over the whole year?
re_all = pv * irr + wt * w
base_all = elec + water * ro_per_m3
deficit_all = np.maximum(0, base_all - re_all)
print(f"\nPeak deficit over year: {np.max(deficit_all):.2f} kW at hour {int(np.argmax(deficit_all))}")
print(f"Hours with deficit > 60 kW: {np.sum(deficit_all > 60)}")
print(f"Hours with deficit > 80 kW: {np.sum(deficit_all > 80)}")
print(f"Hours with deficit > 100 kW: {np.sum(deficit_all > 100)}")
print(f"Peak elec demand: {np.max(elec):.2f} kW")
