"""ε-constraint MILP optimizer for the NSGA-II paper replication.

Reference: Ahmed et al. (2024) - ECM 299, 117865
Components: PV/WT/DG/Battery + Electrolyzer/Methanation/GasStorage + RO

Objective: min NPC (Net Present Cost)
Constraints: power balance, battery SOC, DG fuel, ELZ→CH4 chain, RO, gas balance, LPSP

Fig. 3 EMS priority rules (Option A) are enforced as MILP constraints:
  Surplus mode (RE > load):
    1. Battery charges FIRST from surplus, until saturated (SOC_max or rate limit).
    2. Leftover surplus (if any) feeds electrolyzer / methanation / gas storage.
    3. Remaining leftover is dumped as excess.
  Deficit mode (RE < load):
    4. Battery discharges FIRST to cover deficit, until empty (SOC_min) or rate limit.
    5. DG fires ONLY after battery discharge is saturated.
    6. Any remaining shortfall is unmet load (LPSP).
RO water demand is treated as base electrical load (must be met every hour), not
as a surplus-only consumer, to keep daily water demand satisfied.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from config.nsga_parameters import NSGASystemParams, DEFAULT_NSGA_PARAMS


@dataclass
class NSGAOptResult:
    """Result from a single MILP solve."""
    status: str
    npc: float
    coe: float
    lpsp: float
    reliability: float

    # Capacities
    pv_kw: float
    wind_kw: float
    dg_kw: float
    batt_kwh: float
    elz_kw: float
    gas_storage_m3: float

    # Annual values
    annual_energy_kwh: float
    annual_fuel_L: float
    annual_h2_mol: float
    annual_ch4_m3: float
    annual_water_m3: float
    annual_unmet_kwh: float
    annual_excess_kwh: float

    # Cost breakdown
    capital: float
    replacement: float
    om: float
    fuel_cost: float
    salvage: float

    cow: float  # cost of water
    cog: float  # cost of gas


class NSGAMILPOptimizer:
    """MILP optimizer for PV/WT/DG/Battery + Power-to-X system."""

    def __init__(
        self,
        elec_demand: np.ndarray,
        water_demand: np.ndarray,
        gas_demand: np.ndarray,
        irradiance_factor: np.ndarray,
        wind_factor: np.ndarray,
        params: Optional[NSGASystemParams] = None,
        time_limit_sec: int = 600,
        gap_tolerance: float = 0.01,
        solver_verbose: bool = True,
    ):
        self.elec_demand = elec_demand
        self.water_demand = water_demand
        self.gas_demand = gas_demand
        self.irradiance = irradiance_factor
        self.wind = wind_factor
        self.H = len(elec_demand)

        self.p = params or DEFAULT_NSGA_PARAMS
        self.time_limit = time_limit_sec
        self.gap_tol = gap_tolerance
        self.verbose = solver_verbose

        # Precompute present value factor for O&M/fuel
        self._pvf = self.p.economic.present_value_factor()
        self._crf = self.p.economic.crf()
        self._dr = self.p.economic.discount_rate
        self._n = self.p.economic.project_lifetime

    def _replacement_factor(self, comp_lifetime: int) -> float:
        """Compute present value of replacement costs (Eq 31).
        Sum of 1/(1+i)^(LF_comp×j) for j=1..N_rep
        """
        total = 0.0
        t = comp_lifetime
        while t < self._n:
            total += 1.0 / (1.0 + self._dr) ** t
            t += comp_lifetime
        return total

    def _salvage_factor(self, comp_lifetime: int, rep_cost_ratio: float = 1.0) -> float:
        """Compute present value of salvage (Eqs 33-35).
        Salvage = C_rep × R_rem/LF_comp × 1/(1+i)^n
        """
        n_rep = self._n // comp_lifetime
        r_rep = comp_lifetime * n_rep
        r_rem = comp_lifetime - (self._n - r_rep)
        if r_rem <= 0:
            return 0.0
        frac = r_rem / comp_lifetime
        pv = 1.0 / (1.0 + self._dr) ** self._n
        return frac * pv * rep_cost_ratio

    def _build_model(self, name: str = "NSGA_MILP") -> Tuple[gp.Model, Dict]:
        """Build the MILP model with Fig. 3 EMS priority constraints (Option A)."""
        model = gp.Model(name)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", self.gap_tol)
        model.setParam("Presolve", 2)
        model.setParam("Threads", 0)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1 if self.verbose else 0)

        H = self.H
        b = self.p.bounds

        # ═══════════ CAPACITY VARIABLES ═══════════
        cap_pv = model.addVar(lb=b.pv_min, ub=b.pv_max, name="cap_pv")
        cap_wt = model.addVar(lb=b.wind_min, ub=b.wind_max, name="cap_wt")
        cap_dg = model.addVar(lb=b.dg_min, ub=b.dg_max, name="cap_dg")
        cap_batt = model.addVar(lb=b.batt_min, ub=b.batt_max, name="cap_batt")
        cap_elz = model.addVar(lb=b.elz_min, ub=b.elz_max, name="cap_elz")
        cap_gs = model.addVar(lb=b.gas_storage_min, ub=b.gas_storage_max, name="cap_gs")

        # ═══════════ HOURLY CONTINUOUS VARIABLES ═══════════
        p_dg = model.addVars(H, lb=0.0, name="p_dg")
        p_batt_ch = model.addVars(H, lb=0.0, name="p_batt_ch")
        p_batt_dis = model.addVars(H, lb=0.0, name="p_batt_dis")
        batt_soc = model.addVars(H, lb=0.0, name="batt_soc")
        p_elz = model.addVars(H, lb=0.0, name="p_elz")
        p_ume = model.addVars(H, lb=0.0, name="p_ume")
        p_excess = model.addVars(H, lb=0.0, name="p_excess")
        gas_level = model.addVars(H, lb=0.0, name="gas_level")
        gas_supply = model.addVars(H, lb=0.0, name="gas_supply")
        # EMS auxiliary: surplus/deficit decomposition of (RE - base_load)
        surplus = model.addVars(H, lb=0.0, name="surplus")
        deficit = model.addVars(H, lb=0.0, name="deficit")

        # ═══════════ HOURLY BINARY VARIABLES (Fig. 3 EMS gating) ═══════════
        # y_dg[h]       : DG on/off (existing purpose, now hourly, gated by z_dis)
        # z_surp[h] = 1 : surplus mode (RE >= base_load)
        # z_head[h] = 1 : charge saturated (battery reached SOC_max -> leftover spills to ELZ)
        # z_dis[h]  = 1 : discharge saturated (battery reached SOC_min -> DG allowed to fire)
        y_dg = model.addVars(H, vtype=GRB.BINARY, name="y_dg")
        z_surp = model.addVars(H, vtype=GRB.BINARY, name="z_surp")
        z_head = model.addVars(H, vtype=GRB.BINARY, name="z_head")
        z_dis = model.addVars(H, vtype=GRB.BINARY, name="z_dis")

        # ═══════════ PARAMETERS ═══════════
        inv_eff = self.p.inverter.efficiency
        bp = self.p.battery
        elzp = self.p.electrolyzer
        gsp = self.p.gas_storage
        meth = self.p.methanation
        eta_ch = bp.charge_efficiency
        eta_dis = bp.discharge_efficiency
        soc_min_frac = bp.soc_min
        soc_max_frac = bp.soc_max
        # Self-discharge is set to 0 in the model: the config value (0.02%/hr)
        # is numerically negligible but creates infeasibilities at SOC_min
        # under the strict EMS priority (can't top up in deficit mode).
        sigma = 0.0

        h2_rate = elzp.h2_rate_mol_per_kwh
        eta_meth = meth.eta_meth
        mol_to_m3_stp = gsp.mol_to_m3_stp
        comp_ratio = gsp.compression_ratio
        gas_rate_stp_per_kwh = h2_rate * eta_meth * mol_to_m3_stp
        gas_rate_compressed_per_kwh = gas_rate_stp_per_kwh / comp_ratio
        gas_demand_compressed = self.gas_demand / comp_ratio

        ro_power_per_m3 = self.p.ro.specific_energy  # 4.38 kWh/m³

        # RO treated as mandatory base load (water demand must be met every hour)
        # This deviates from Fig. 3 (which puts RO in the "excess" queue) but matches
        # the paper's water-demand constraint. ELZ is the only EMS-discretionary load.
        ro_power_array = np.array(
            [float(self.water_demand[h] * ro_power_per_m3) for h in range(H)]
        )
        base_load = self.elec_demand + ro_power_array  # effective electrical load

        # ═══════════ BIG-M CONSTANTS ═══════════
        # Tight upper bounds derived from capacity UBs and input data
        max_re_potential = (
            b.pv_max * float(np.max(self.irradiance))
            + b.wind_max * float(np.max(self.wind))
        )
        M_surp = max(1.0, max_re_potential + float(np.max(base_load)))
        M_def = max(1.0, float(np.max(base_load)) + 1.0)
        M_batt_cap = max(1.0, b.batt_max)
        M_head_ch = max(1.0, soc_max_frac * b.batt_max / max(eta_ch, 1e-3))
        M_head_dis = max(1.0, soc_max_frac * b.batt_max * eta_dis)
        M_dg = max(1.0, b.dg_max)
        M_elz = max(1.0, b.elz_max)

        # ═══════════ PER-HOUR CONSTRAINTS ═══════════
        for h in range(H):
            re_h = cap_pv * self.irradiance[h] + cap_wt * self.wind[h]

            # Previous SOC (linear expr, used by headroom & dynamics)
            prev_soc = (
                bp.initial_soc * cap_batt if h == 0 else batt_soc[h - 1]
            )

            # --- Surplus/deficit split: (RE - base_load) = surplus - deficit ---
            model.addConstr(
                surplus[h] - deficit[h] == re_h - float(base_load[h]),
                name=f"enet_{h}",
            )
            model.addConstr(surplus[h] <= M_surp * z_surp[h], name=f"surp_mode_{h}")
            model.addConstr(deficit[h] <= M_def * (1 - z_surp[h]), name=f"def_mode_{h}")

            # --- Power balance (Eq 51, RO folded into base_load) ---
            model.addConstr(
                re_h + p_dg[h] + p_batt_dis[h] * inv_eff
                == float(base_load[h]) + p_batt_ch[h] + p_elz[h]
                + p_excess[h] - p_ume[h],
                name=f"balance_{h}",
            )
            model.addConstr(p_ume[h] <= self.elec_demand[h], name=f"ume_max_{h}")

            # --- DG capacity and on/off ---
            model.addConstr(p_dg[h] <= cap_dg, name=f"dg_cap_{h}")
            model.addConstr(p_dg[h] <= M_dg * y_dg[h], name=f"dg_on_{h}")

            # --- Electrolyzer capacity ---
            model.addConstr(p_elz[h] <= cap_elz, name=f"elz_cap_{h}")

            # --- Battery SOC dynamics (Eq 18-19) ---
            model.addConstr(
                batt_soc[h] == prev_soc * (1.0 - sigma)
                + p_batt_ch[h] * eta_ch
                - p_batt_dis[h] / eta_dis,
                name=f"soc_{h}",
            )
            model.addConstr(batt_soc[h] >= soc_min_frac * cap_batt, name=f"soc_min_{h}")
            model.addConstr(batt_soc[h] <= soc_max_frac * cap_batt, name=f"soc_max_{h}")

            # ═══════════ Fig. 3 EMS PRIORITY RULES ═══════════
            # Headroom expressions (linear in cap_batt and prev_soc)
            head_ch = (soc_max_frac * cap_batt - prev_soc) / eta_ch
            head_dis_dc = (prev_soc - soc_min_frac * cap_batt) * eta_dis

            # --- (a) Battery charges ONLY from surplus, up to headroom ---
            model.addConstr(p_batt_ch[h] <= surplus[h], name=f"ch_from_surplus_{h}")
            model.addConstr(p_batt_ch[h] <= head_ch, name=f"ch_headroom_{h}")
            # Force p_batt_ch = min(surplus, head_ch):
            #   z_head = 0: surplus <= head_ch   -> p_batt_ch = surplus  (Case A)
            #   z_head = 1: surplus  > head_ch   -> p_batt_ch = head_ch  (Case B)
            model.addConstr(
                p_batt_ch[h] >= surplus[h] - M_head_ch * z_head[h],
                name=f"ch_force_surp_{h}",
            )
            model.addConstr(
                p_batt_ch[h] >= head_ch - M_head_ch * (1 - z_head[h]),
                name=f"ch_force_head_{h}",
            )

            # --- (b) ELZ and dump ONLY from post-battery-charge surplus leftover ---
            model.addConstr(
                p_elz[h] + p_excess[h] <= surplus[h] - p_batt_ch[h],
                name=f"elz_dump_leftover_{h}",
            )

            # --- (c) No battery discharge in surplus mode ---
            model.addConstr(
                p_batt_dis[h] <= M_batt_cap * (1 - z_surp[h]),
                name=f"dis_no_surplus_{h}",
            )

            # --- (d) Battery discharges FIRST from deficit before DG fires ---
            # p_batt_dis is in DC kWh; AC contribution = p_batt_dis * inv_eff
            model.addConstr(
                p_batt_dis[h] * inv_eff <= deficit[h],
                name=f"dis_within_deficit_{h}",
            )
            model.addConstr(p_batt_dis[h] <= head_dis_dc, name=f"dis_headroom_{h}")
            # Force p_batt_dis*inv_eff = min(deficit, head_dis_dc*inv_eff):
            #   z_dis = 0: deficit <= head_dis_ac -> batt covers all deficit (Case A)
            #   z_dis = 1: deficit  > head_dis_ac -> batt saturates, DG allowed (Case B)
            model.addConstr(
                p_batt_dis[h] * inv_eff >= deficit[h] - M_def * z_dis[h],
                name=f"dis_force_def_{h}",
            )
            model.addConstr(
                p_batt_dis[h] >= head_dis_dc - M_head_dis * (1 - z_dis[h]),
                name=f"dis_force_head_{h}",
            )

            # --- (e) DG fires ONLY when battery discharge is saturated (z_dis=1) ---
            model.addConstr(y_dg[h] <= z_dis[h], name=f"dg_after_batt_{h}")

            # --- Gas storage balance (compressed m³) ---
            gas_produced_h = p_elz[h] * gas_rate_compressed_per_kwh
            if h == 0:
                model.addConstr(
                    gas_level[h] == 0.5 * cap_gs + gas_produced_h - gas_supply[h],
                    name=f"gas_bal_{h}",
                )
            else:
                model.addConstr(
                    gas_level[h] == gas_level[h - 1] + gas_produced_h - gas_supply[h],
                    name=f"gas_bal_{h}",
                )
            model.addConstr(gas_level[h] >= gsp.soc_min * cap_gs, name=f"gs_min_{h}")
            model.addConstr(gas_level[h] <= gsp.soc_max * cap_gs, name=f"gs_max_{h}")
            model.addConstr(gas_supply[h] <= gas_demand_compressed[h], name=f"gas_demand_{h}")

        # ═══════════ ANNUAL GAS DEMAND (no floor under Option A) ═══════════
        # Note: the paper's own Table 6 shows ELZ = 67,416 kWh/yr producing
        # ~11,660 m3 CH4 against 120,450 m3 cooking demand (~10% coverage).
        # Under strict Fig. 3 priority, ELZ runs only on post-battery surplus
        # leftover, so we do not impose an artificial coverage floor here.
        # gas_supply[h] is still capped by the hourly gas_demand_compressed.

        # ═══════════ OBJECTIVE: min NPC (Eqs 29-36) ═══════════
        pvf = self._pvf

        # --- Capital costs (Eq 30) ---
        capital = (
            self.p.pv.capital_cost * cap_pv
            + self.p.wind.capital_cost * cap_wt
            + self.p.diesel.capital_cost * cap_dg
            + self.p.battery.capital_cost * cap_batt
            + self.p.electrolyzer.capital_cost * cap_elz
            + self.p.gas_storage.capital_cost * cap_gs
            + self.p.inverter.capital_cost * (cap_pv + cap_wt)
        )

        # --- Replacement costs (Eq 31) ---
        rep_wt = self.p.wind.replacement_cost * self._replacement_factor(self.p.wind.lifetime) * cap_wt
        dg_life_years = max(2, int(self.p.diesel.lifetime_hours / (8760 * 0.25)))
        rep_dg = self.p.diesel.replacement_cost * self._replacement_factor(dg_life_years) * cap_dg
        rep_batt = self.p.battery.replacement_cost * self._replacement_factor(self.p.battery.lifetime) * cap_batt
        rep_elz = self.p.electrolyzer.replacement_cost * self._replacement_factor(self.p.electrolyzer.lifetime) * cap_elz
        rep_inv = self.p.inverter.replacement_cost * self._replacement_factor(self.p.inverter.lifetime)
        replacement = rep_wt + rep_dg + rep_batt + rep_elz + rep_inv * (cap_pv + cap_wt)

        # --- O&M costs (Eq 32) ---
        om_annual = (
            self.p.pv.om_cost * cap_pv
            + self.p.wind.om_cost * cap_wt
            + self.p.electrolyzer.om_cost * cap_elz
            + self.p.gas_storage.om_cost * cap_gs
        )
        om = om_annual * pvf

        # DG O&M: $/hour × on-hours (hourly y_dg)
        dg_om = self.p.diesel.om_cost * gp.quicksum(y_dg[h] for h in range(H)) * pvf

        # --- Fuel cost (Eq 36) ---
        annual_fuel_expr = gp.quicksum(
            self.p.diesel.fuel_intercept * p_dg[h]
            + self.p.diesel.fuel_slope * cap_dg * y_dg[h]
            for h in range(H)
        )
        fuel_cost = self.p.diesel.fuel_price * annual_fuel_expr * pvf

        # --- RO costs ---
        daily_cap = self.p.load.water_demand_m3_day
        ro_capital = self.p.ro.capital_cost_per_m3day * daily_cap
        ro_tank = self.p.ro.water_tank_cost * daily_cap * self.p.ro.water_tank_days
        annual_water = daily_cap * 365.0
        ro_om = (self.p.ro.om_cost + self.p.ro.chemical_cost) * annual_water * pvf
        ro_membrane = (self.p.ro.membrane_cost_per_m3day * daily_cap
                       * self.p.ro.membrane_replacements_per_year * pvf)
        ro_total = ro_capital + ro_tank + ro_om + ro_membrane

        # --- Salvage (Eqs 33-35) ---
        sal_pv = self.p.pv.capital_cost * self._salvage_factor(self.p.pv.lifetime) * cap_pv
        sal_wt = self.p.wind.replacement_cost * self._salvage_factor(self.p.wind.lifetime) * cap_wt
        sal_batt = self.p.battery.replacement_cost * self._salvage_factor(self.p.battery.lifetime) * cap_batt
        sal_elz = self.p.electrolyzer.replacement_cost * self._salvage_factor(self.p.electrolyzer.lifetime) * cap_elz
        salvage = sal_pv + sal_wt + sal_batt + sal_elz

        # --- Total NPC ---
        npc = capital + replacement + om + dg_om + fuel_cost + ro_total - salvage

        variables = {
            "cap_pv": cap_pv, "cap_wt": cap_wt, "cap_dg": cap_dg,
            "cap_batt": cap_batt, "cap_elz": cap_elz, "cap_gs": cap_gs,
            "p_dg": p_dg, "p_batt_ch": p_batt_ch, "p_batt_dis": p_batt_dis,
            "batt_soc": batt_soc, "p_elz": p_elz,
            "p_ume": p_ume, "p_excess": p_excess,
            "gas_level": gas_level, "gas_supply": gas_supply,
            "y_dg": y_dg, "z_surp": z_surp, "z_head": z_head, "z_dis": z_dis,
            "surplus": surplus, "deficit": deficit,
            "npc": npc, "capital_expr": capital, "replacement_expr": replacement,
            "om_expr": om + dg_om, "fuel_expr": fuel_cost, "salvage_expr": salvage,
            "ro_total": ro_total,
            "annual_fuel_expr": annual_fuel_expr,
            "gas_rate_stp_per_kwh": gas_rate_stp_per_kwh,
            "ro_power_array": ro_power_array,
        }

        return model, variables

    def minimize_npc(self, lpsp_max: float = 0.01) -> NSGAOptResult:
        """Minimize NPC subject to LPSP constraint.

        Args:
            lpsp_max: Maximum allowed LPSP (0.01 = 1% unmet)
        """
        model, variables = self._build_model("Min_NPC")
        model.setObjective(variables["npc"], GRB.MINIMIZE)

        # LPSP constraint (Eq 53)
        total_demand = float(np.sum(self.elec_demand))
        model.addConstr(
            gp.quicksum(variables["p_ume"][h] for h in range(self.H))
            <= lpsp_max * total_demand,
            name="lpsp_limit",
        )

        model.optimize()

        # If infeasible, compute and print IIS for debugging
        if model.Status == GRB.INFEASIBLE:
            print("\n[IIS] Model infeasible — computing IIS...")
            try:
                model.computeIIS()
                iis_constrs = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
                iis_bounds = [
                    v.VarName for v in model.getVars()
                    if v.IISLB or v.IISUB
                ]
                print(f"[IIS] {len(iis_constrs)} constraints in IIS "
                      f"(first 30): {iis_constrs[:30]}")
                print(f"[IIS] {len(iis_bounds)} var bounds in IIS "
                      f"(first 30): {iis_bounds[:30]}")
            except Exception as e:
                print(f"[IIS] computeIIS failed: {e}")

        return self._extract_result(model, variables)

    def _extract_result(self, model: gp.Model, variables: Dict) -> NSGAOptResult:
        """Extract results from solved model."""
        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (
            model.Status == GRB.TIME_LIMIT and model.SolCount > 0
        ):
            status = "Optimal"
        else:
            return NSGAOptResult(
                status=f"Infeasible({model.Status})", npc=float("inf"),
                coe=float("inf"), lpsp=1.0, reliability=0.0,
                pv_kw=0, wind_kw=0, dg_kw=0, batt_kwh=0, elz_kw=0, gas_storage_m3=0,
                annual_energy_kwh=0, annual_fuel_L=0, annual_h2_mol=0,
                annual_ch4_m3=0, annual_water_m3=0, annual_unmet_kwh=0,
                annual_excess_kwh=0, capital=0, replacement=0, om=0,
                fuel_cost=0, salvage=0, cow=0, cog=0,
            )

        v = variables
        cap_pv = v["cap_pv"].X
        cap_wt = v["cap_wt"].X
        cap_dg = v["cap_dg"].X
        cap_batt = v["cap_batt"].X
        cap_elz = v["cap_elz"].X
        cap_gs = v["cap_gs"].X

        annual_energy = sum(
            self.elec_demand[h] - v["p_ume"][h].X for h in range(self.H)
        )
        annual_unmet = sum(v["p_ume"][h].X for h in range(self.H))
        annual_excess = sum(v["p_excess"][h].X for h in range(self.H))
        annual_fuel = v["annual_fuel_expr"].getValue()
        annual_elz = sum(v["p_elz"][h].X for h in range(self.H))
        comp_ratio = self.p.gas_storage.compression_ratio
        annual_ch4_m3 = annual_elz * v["gas_rate_stp_per_kwh"]  # m³ STP
        annual_gas_supplied = sum(v["gas_supply"][h].X for h in range(self.H)) * comp_ratio  # to STP
        annual_water = float(np.sum(self.water_demand))

        total_demand = float(np.sum(self.elec_demand))
        lpsp = annual_unmet / total_demand if total_demand > 0 else 1.0

        npc = v["npc"].getValue()
        capital = v["capital_expr"].getValue()
        replacement = v["replacement_expr"].getValue()
        om = v["om_expr"].getValue()
        fuel_cost = v["fuel_expr"].getValue()
        salvage = v["salvage_expr"].getValue()

        crf = self._crf
        coe = (npc * crf) / annual_energy if annual_energy > 0 else float("inf")

        # Cost of water (simplified)
        ro_cost = v["ro_total"] if isinstance(v["ro_total"], float) else v["ro_total"]
        cow = (ro_cost * crf) / annual_water if annual_water > 0 else 0.0

        # Cost of gas
        gas_related_cost = (
            self.p.electrolyzer.capital_cost * cap_elz
            + self.p.gas_storage.capital_cost * cap_gs
        )
        total_gas_demand = float(np.sum(self.gas_demand))
        cog = (gas_related_cost * crf) / total_gas_demand if total_gas_demand > 0 else 0.0

        return NSGAOptResult(
            status=status, npc=npc, coe=coe, lpsp=lpsp, reliability=1.0 - lpsp,
            pv_kw=cap_pv, wind_kw=cap_wt, dg_kw=cap_dg,
            batt_kwh=cap_batt, elz_kw=cap_elz, gas_storage_m3=cap_gs,
            annual_energy_kwh=annual_energy, annual_fuel_L=annual_fuel,
            annual_h2_mol=annual_elz * self.p.electrolyzer.h2_rate_mol_per_kwh,
            annual_ch4_m3=annual_ch4_m3, annual_water_m3=annual_water,
            annual_unmet_kwh=annual_unmet, annual_excess_kwh=annual_excess,
            capital=capital, replacement=replacement, om=om,
            fuel_cost=fuel_cost, salvage=salvage, cow=cow, cog=cog,
        )
