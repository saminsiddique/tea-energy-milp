"""Rule-based energy management simulator implementing paper Fig. 3.

Reference: Ahmed et al. (2024) — ECM 299, 117865
Section 2.7.3 "Energy management strategy" and Fig. 3.

Hour-by-hour dispatch (no optimization in the inner loop):

  Surplus mode (RE > base_load):
      1. Charge battery first (up to SOC_max, rate-limited).
      2. Feed electrolyzer with whatever leftover remains (up to ELZ cap).
      3. Dump the rest as excess.

  Deficit mode (RE < base_load):
      1. Discharge battery first (down to SOC_min, rate-limited).
      2. Fire diesel generator for any remaining shortfall.
      3. If still short, record as unmet (LPSP).

Water demand is treated as hourly-pinned base load (RO power folded into
base_load), so water is always met as long as LPSP is satisfied. Gas demand is
served from the methanated CH4 in the compressed gas tank; any uncovered gas
demand is simply reported (no constraint).

After the loop, NPC / COE / COW / COG are computed per paper Eqs 29–44.

The inner loop is @njit-compiled for speed (~1 ms per full-year simulation
after the first call, which pays the JIT compile cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numba import njit

from config.nsga_parameters import NSGASystemParams


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class SimulationResult:
    """Result of a single rule-based EMS simulation + cost computation."""
    # Echoed sizes
    pv_kw: float
    wind_kw: float
    dg_kw: float
    batt_kwh: float
    elz_kw: float
    gas_storage_m3: float
    # Cost breakdown ($)
    capital: float
    replacement: float
    om: float
    fuel_cost: float
    salvage: float
    ro_total: float
    npc: float
    # Levelized costs
    coe: float  # $/kWh
    cow: float  # $/m3 water
    cog: float  # $/m3 gas
    # Dispatch annuals
    annual_energy_kwh: float  # delivered electrical energy (load - unmet)
    annual_unmet_kwh: float
    annual_excess_kwh: float
    annual_fuel_L: float
    annual_ch4_m3_stp: float  # methane produced (supply side, STP)
    annual_ch4_delivered_stp: float  # methane drawn from storage (demand side, STP)
    annual_dg_hours: int
    lpsp: float  # fraction unmet
    renewable_fraction: float  # 1 - non-renewable fraction


# ============================================================
# Hot inner loop (numba JIT)
# ============================================================

@njit(cache=True, fastmath=True)
def _simulate_dispatch(
    # Time series inputs (length H)
    elec_demand: np.ndarray,
    ro_power: np.ndarray,
    gas_demand_comp: np.ndarray,  # compressed m3/h
    irradiance: np.ndarray,
    wind: np.ndarray,
    # Sizes
    pv_kw: float,
    wind_kw: float,
    dg_kw: float,
    batt_kwh: float,
    elz_kw: float,
    gas_storage_m3: float,
    # Battery params
    soc_min_frac: float,
    soc_max_frac: float,
    initial_soc_frac: float,
    eta_ch: float,
    eta_dis: float,
    # Inverter
    inv_eff: float,
    # Gas
    gas_rate_stp_per_kwh: float,
    gas_rate_comp_per_kwh: float,
    gs_soc_min_frac: float,
    gs_soc_max_frac: float,
):
    """Rule-based hour-by-hour dispatch. Returns annual aggregates.

    All parameters are floats/arrays (numba-compatible).
    """
    H = elec_demand.shape[0]

    batt_soc = initial_soc_frac * batt_kwh
    batt_min = soc_min_frac * batt_kwh
    batt_max = soc_max_frac * batt_kwh

    # Gas tank starts EMPTY: "delivered" CH4 must come from production,
    # otherwise the optimizer exploits free initial inventory.
    gas_level = 0.0  # compressed m3
    gas_min = 0.0
    gas_max = gs_soc_max_frac * gas_storage_m3

    annual_unmet = 0.0
    annual_excess = 0.0
    annual_fuel_L = 0.0
    annual_dg_hours = 0
    annual_ch4_prod_stp = 0.0
    annual_ch4_delivered_stp = 0.0
    annual_energy_served = 0.0
    annual_dg_energy = 0.0

    # DG fuel curve constants (Eq 21): F0=0.246, F1=0.08145
    F0 = 0.246
    F1 = 0.08145

    for h in range(H):
        re_h = pv_kw * irradiance[h] + wind_kw * wind[h]
        load_h = elec_demand[h] + ro_power[h]  # base load includes RO
        e_net = re_h - load_h

        p_elz_h = 0.0
        excess_h = 0.0
        dg_out_h = 0.0
        unmet_h = 0.0

        if e_net >= 0.0:
            # ───── Surplus branch ─────
            surplus = e_net

            # (1) Charge battery first
            ch_room_kwh = (batt_max - batt_soc) / max(eta_ch, 1e-9)
            # Rate limit: 1C (full capacity in 1 hour)
            ch_rate_limit = batt_kwh
            p_charge = surplus
            if p_charge > ch_room_kwh:
                p_charge = ch_room_kwh
            if p_charge > ch_rate_limit:
                p_charge = ch_rate_limit
            if p_charge < 0.0:
                p_charge = 0.0
            batt_soc += p_charge * eta_ch
            surplus -= p_charge

            # (2) Feed electrolyzer with what's left (up to ELZ cap)
            p_elz_h = surplus
            if p_elz_h > elz_kw:
                p_elz_h = elz_kw
            if p_elz_h < 0.0:
                p_elz_h = 0.0
            surplus -= p_elz_h

            # Methanation + gas storage (compressed volume)
            ch4_prod_stp = p_elz_h * gas_rate_stp_per_kwh
            ch4_prod_comp = p_elz_h * gas_rate_comp_per_kwh
            gas_level += ch4_prod_comp
            if gas_level > gas_max:
                # tank overflow -> untracked vent
                gas_level = gas_max
            annual_ch4_prod_stp += ch4_prod_stp

            # (3) Dump the rest
            excess_h = surplus if surplus > 0.0 else 0.0

        else:
            # ───── Deficit branch ─────
            needed = -e_net  # kWh (AC side)

            # (1) Discharge battery first
            dis_room_kwh_dc = (batt_soc - batt_min)
            dis_rate_limit_dc = batt_kwh  # 1C
            # AC energy deliverable
            dis_room_ac = dis_room_kwh_dc * eta_dis * inv_eff
            dis_rate_ac = dis_rate_limit_dc * eta_dis * inv_eff
            p_dis_ac = needed
            if p_dis_ac > dis_room_ac:
                p_dis_ac = dis_room_ac
            if p_dis_ac > dis_rate_ac:
                p_dis_ac = dis_rate_ac
            if p_dis_ac < 0.0:
                p_dis_ac = 0.0
            # Update SOC from AC delivered
            p_dis_dc = p_dis_ac / (eta_dis * inv_eff) if (eta_dis * inv_eff) > 0 else 0.0
            batt_soc -= p_dis_dc
            if batt_soc < batt_min:
                batt_soc = batt_min
            needed -= p_dis_ac

            # (2) Diesel generator
            if needed > 0.0:
                dg_out_h = needed
                if dg_out_h > dg_kw:
                    dg_out_h = dg_kw
                needed -= dg_out_h
                if dg_out_h > 0.0:
                    annual_dg_hours += 1
                    # Fuel consumption (Eq 21): F0 * P_dg + F1 * P_rated
                    annual_fuel_L += F0 * dg_out_h + F1 * dg_kw
                    annual_dg_energy += dg_out_h

            # (3) Unmet
            if needed > 0.0:
                unmet_h = needed

        # ───── Gas demand draw (any hour) ─────
        gas_req_comp = gas_demand_comp[h]
        if gas_req_comp > 0.0:
            available = gas_level - gas_min
            if available < 0.0:
                available = 0.0
            gas_draw_comp = gas_req_comp
            if gas_draw_comp > available:
                gas_draw_comp = available
            gas_level -= gas_draw_comp
            # Back to STP for reporting
            annual_ch4_delivered_stp += gas_draw_comp * (gas_rate_stp_per_kwh / max(gas_rate_comp_per_kwh, 1e-12))

        # Accumulate annual energy served (load actually met)
        served_h = load_h - unmet_h
        if served_h < 0.0:
            served_h = 0.0
        annual_energy_served += served_h
        annual_unmet += unmet_h
        annual_excess += excess_h

    return (
        annual_unmet,
        annual_excess,
        annual_fuel_L,
        float(annual_dg_hours),
        annual_ch4_prod_stp,
        annual_ch4_delivered_stp,
        annual_energy_served,
        annual_dg_energy,
    )


# ============================================================
# Cost helpers (paper Eqs 29–44)
# ============================================================

def _replacement_factor(dr: float, comp_life: float, project_life: int) -> float:
    """Sum of 1/(1+i)^(LF_comp * j) for j=1..n_rep (Eq 31)."""
    if comp_life <= 0:
        return 0.0
    total = 0.0
    t = comp_life
    while t < project_life:
        total += 1.0 / (1.0 + dr) ** t
        t += comp_life
    return total


def _salvage_factor(dr: float, comp_life: float, project_life: int) -> float:
    """PV of end-of-project salvage (Eqs 33–35)."""
    if comp_life <= 0:
        return 0.0
    n_rep = int(project_life // comp_life)
    r_rep = comp_life * n_rep
    r_rem = comp_life - (project_life - r_rep)
    if r_rem <= 0:
        return 0.0
    frac = r_rem / comp_life
    pv = 1.0 / (1.0 + dr) ** project_life
    return frac * pv


# ============================================================
# Public entry point
# ============================================================

def simulate(
    sizes: Dict[str, float],
    data: Dict[str, np.ndarray],
    params: NSGASystemParams,
) -> SimulationResult:
    """Run Fig. 3 EMS simulation for one sizing and return all metrics."""
    pv_kw = float(sizes["pv_kw"])
    wind_kw = float(sizes["wind_kw"])
    dg_kw = float(sizes["dg_kw"])
    batt_kwh = float(sizes["batt_kwh"])
    elz_kw = float(sizes["elz_kw"])
    gas_storage_m3 = float(sizes["gas_storage_m3"])

    elec_demand = np.asarray(data["elec_demand"], dtype=np.float64)
    water_demand = np.asarray(data["water_demand"], dtype=np.float64)
    gas_demand = np.asarray(data["gas_demand"], dtype=np.float64)
    irradiance = np.asarray(data["irradiance_factor"], dtype=np.float64)
    wind = np.asarray(data["wind_factor"], dtype=np.float64)

    # Derived arrays
    ro_power = water_demand * params.ro.specific_energy  # kWh/h
    comp_ratio = params.gas_storage.compression_ratio
    gas_demand_comp = gas_demand / comp_ratio  # compressed m3/h

    # Gas production rate (Eq 5): mol H2 → mol CH4 → m3 STP → compressed
    h2_rate = params.electrolyzer.h2_rate_mol_per_kwh
    eta_meth = params.methanation.eta_meth
    mol_to_m3_stp = params.gas_storage.mol_to_m3_stp
    gas_rate_stp_per_kwh = h2_rate * eta_meth * mol_to_m3_stp
    gas_rate_comp_per_kwh = gas_rate_stp_per_kwh / comp_ratio

    (
        annual_unmet,
        annual_excess,
        annual_fuel_L,
        annual_dg_hours_f,
        annual_ch4_prod_stp,
        annual_ch4_delivered_stp,
        annual_energy_served,
        annual_dg_energy,
    ) = _simulate_dispatch(
        elec_demand=elec_demand,
        ro_power=ro_power,
        gas_demand_comp=gas_demand_comp,
        irradiance=irradiance,
        wind=wind,
        pv_kw=pv_kw,
        wind_kw=wind_kw,
        dg_kw=dg_kw,
        batt_kwh=batt_kwh,
        elz_kw=elz_kw,
        gas_storage_m3=gas_storage_m3,
        soc_min_frac=params.battery.soc_min,
        soc_max_frac=params.battery.soc_max,
        initial_soc_frac=params.battery.initial_soc,
        eta_ch=params.battery.charge_efficiency,
        eta_dis=params.battery.discharge_efficiency,
        inv_eff=params.inverter.efficiency,
        gas_rate_stp_per_kwh=gas_rate_stp_per_kwh,
        gas_rate_comp_per_kwh=gas_rate_comp_per_kwh,
        gs_soc_min_frac=params.gas_storage.soc_min,
        gs_soc_max_frac=params.gas_storage.soc_max,
    )
    annual_dg_hours = int(annual_dg_hours_f)

    # ───── Cost model (paper Eqs 29–44) ─────
    eco = params.economic
    dr = eco.discount_rate
    n = eco.project_lifetime
    pvf = eco.present_value_factor()
    crf = eco.crf()

    # (1) Initial capital (Eq 30) — methanation sized equal to electrolyzer
    meth_kw = elz_kw  # Paper §2.4.2
    inv_kw = pv_kw + wind_kw  # inverter sized to RE peak
    capital = (
        params.pv.capital_cost * pv_kw
        + params.wind.capital_cost * wind_kw
        + params.diesel.capital_cost * dg_kw
        + params.battery.capital_cost * batt_kwh
        + params.electrolyzer.capital_cost * elz_kw
        + params.methanation.capital_cost * meth_kw
        + params.gas_storage.capital_cost * gas_storage_m3
        + params.inverter.capital_cost * inv_kw
    )

    # (2) Replacement (Eq 31). PV lifetime = project lifetime → no PV rep.
    rep = 0.0
    rep += params.wind.replacement_cost * _replacement_factor(dr, params.wind.lifetime, n) * wind_kw
    # DG replacement based on actual operating hours (paper 15,000 h lifetime)
    dg_life_hours = params.diesel.lifetime_hours
    if annual_dg_hours > 0:
        dg_life_years = dg_life_hours / annual_dg_hours
    else:
        dg_life_years = float(n + 1)  # never replaced
    rep += params.diesel.replacement_cost * _replacement_factor(dr, dg_life_years, n) * dg_kw
    rep += params.battery.replacement_cost * _replacement_factor(dr, params.battery.lifetime, n) * batt_kwh
    rep += params.electrolyzer.replacement_cost * _replacement_factor(dr, params.electrolyzer.lifetime, n) * elz_kw
    rep += params.methanation.replacement_cost * _replacement_factor(dr, params.methanation.lifetime, n) * meth_kw
    rep += params.gas_storage.replacement_cost * _replacement_factor(dr, params.gas_storage.lifetime, n) * gas_storage_m3
    rep += params.inverter.replacement_cost * _replacement_factor(dr, params.inverter.lifetime, n) * inv_kw

    # (3) O&M (Eq 32) — PVF-discounted annual cost × size
    om_annual = (
        params.pv.om_cost * pv_kw
        + params.wind.om_cost * wind_kw
        + params.electrolyzer.om_cost * elz_kw
        + params.methanation.om_cost * meth_kw
        + params.gas_storage.om_cost * gas_storage_m3
        + params.inverter.om_cost * inv_kw
    )
    dg_om_annual = params.diesel.om_cost * annual_dg_hours  # $/h × hours
    om = (om_annual + dg_om_annual) * pvf

    # (4) Fuel (Eq 36) — price × annual_L × PVF
    fuel_cost = params.diesel.fuel_price * annual_fuel_L * pvf

    # (5) Salvage (Eqs 33–35) — positive number, subtracted from NPC
    sal = 0.0
    sal += params.pv.capital_cost * _salvage_factor(dr, params.pv.lifetime, n) * pv_kw
    sal += params.wind.replacement_cost * _salvage_factor(dr, params.wind.lifetime, n) * wind_kw
    sal += params.battery.replacement_cost * _salvage_factor(dr, params.battery.lifetime, n) * batt_kwh
    sal += params.electrolyzer.replacement_cost * _salvage_factor(dr, params.electrolyzer.lifetime, n) * elz_kw
    sal += params.methanation.replacement_cost * _salvage_factor(dr, params.methanation.lifetime, n) * meth_kw
    sal += params.gas_storage.replacement_cost * _salvage_factor(dr, params.gas_storage.lifetime, n) * gas_storage_m3
    sal += params.inverter.replacement_cost * _salvage_factor(dr, params.inverter.lifetime, n) * inv_kw
    sal += params.diesel.replacement_cost * _salvage_factor(dr, dg_life_years, n) * dg_kw

    # (6) RO unit (Eqs 37–42) — CRF-annualized form per paper
    ro = params.ro
    daily_cap = params.load.water_demand_m3_day
    annual_water = daily_cap * 365.0
    # Paper equations 38-42 use CRF; the PVs end up at:
    ro_capital_pv = ro.capital_cost_per_m3day * daily_cap  # Eq 38 minus CRF
    ro_tank_pv = ro.water_tank_cost * daily_cap * ro.water_tank_days  # Eq 42 minus CRF
    ro_om_pv = (ro.om_cost + ro.chemical_cost) * annual_water * pvf  # Eqs 39, 41
    ro_membrane_pv = (
        ro.membrane_cost_per_m3day * daily_cap
        * ro.membrane_replacements_per_year * pvf
    )  # Eq 40
    ro_total = ro_capital_pv + ro_tank_pv + ro_om_pv + ro_membrane_pv

    # (7) Total NPC
    npc = capital + rep + om + fuel_cost + ro_total - sal

    # (8) Levelized costs (Eq 44 and analogues)
    coe = (npc * crf) / annual_energy_served if annual_energy_served > 0 else float("inf")
    cow = (ro_total * crf) / annual_water if annual_water > 0 else 0.0
    # COG: capital/O&M of gas-producing chain, annualized per m3 STP delivered
    gas_chain_pv = (
        params.electrolyzer.capital_cost * elz_kw
        + params.methanation.capital_cost * meth_kw
        + params.gas_storage.capital_cost * gas_storage_m3
        + (params.electrolyzer.om_cost * elz_kw
           + params.methanation.om_cost * meth_kw
           + params.gas_storage.om_cost * gas_storage_m3) * pvf
    )
    total_gas_demand_stp = float(np.sum(gas_demand))
    cog = (gas_chain_pv * crf) / total_gas_demand_stp if total_gas_demand_stp > 0 else 0.0

    total_elec_demand = float(np.sum(elec_demand + ro_power))
    lpsp = annual_unmet / total_elec_demand if total_elec_demand > 0 else 1.0
    renewable_fraction = (
        1.0 - annual_dg_energy / annual_energy_served
        if annual_energy_served > 0 else 0.0
    )

    return SimulationResult(
        pv_kw=pv_kw, wind_kw=wind_kw, dg_kw=dg_kw,
        batt_kwh=batt_kwh, elz_kw=elz_kw, gas_storage_m3=gas_storage_m3,
        capital=capital, replacement=rep, om=om,
        fuel_cost=fuel_cost, salvage=sal, ro_total=ro_total,
        npc=npc, coe=coe, cow=cow, cog=cog,
        annual_energy_kwh=annual_energy_served,
        annual_unmet_kwh=annual_unmet,
        annual_excess_kwh=annual_excess,
        annual_fuel_L=annual_fuel_L,
        annual_ch4_m3_stp=annual_ch4_prod_stp,
        annual_ch4_delivered_stp=annual_ch4_delivered_stp,
        annual_dg_hours=annual_dg_hours,
        lpsp=lpsp,
        renewable_fraction=renewable_fraction,
    )
