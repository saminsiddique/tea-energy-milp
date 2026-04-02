"""ε-constraint MILP optimizer for the NSGA-II paper replication.

Reference: Ahmed et al. (2024) - ECM 299, 117865
Components: PV/WT/DG/Battery + Electrolyzer/Methanation/GasStorage + RO

Objective: min NPC (Net Present Cost)
Constraints: power balance, battery SOC, DG fuel, ELZ→CH4 chain, RO, gas balance, LPSP
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
        """Build the MILP model."""
        model = gp.Model(name)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", self.gap_tol)
        model.setParam("Presolve", 2)
        model.setParam("Threads", 0)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1 if self.verbose else 0)

        H = self.H
        b = self.p.bounds
        BLOCK = 4  # 4-hour blocks for binaries
        n_blocks = (H + BLOCK - 1) // BLOCK

        # ═══════════ CAPACITY VARIABLES ═══════════
        cap_pv = model.addVar(lb=b.pv_min, ub=b.pv_max, name="cap_pv")
        cap_wt = model.addVar(lb=b.wind_min, ub=b.wind_max, name="cap_wt")
        cap_dg = model.addVar(lb=b.dg_min, ub=b.dg_max, name="cap_dg")
        cap_batt = model.addVar(lb=b.batt_min, ub=b.batt_max, name="cap_batt")
        cap_elz = model.addVar(lb=b.elz_min, ub=b.elz_max, name="cap_elz")
        cap_gs = model.addVar(lb=b.gas_storage_min, ub=b.gas_storage_max, name="cap_gs")

        # ═══════════ HOURLY VARIABLES ═══════════
        p_dg = model.addVars(H, lb=0.0, name="p_dg")
        p_batt_ch = model.addVars(H, lb=0.0, name="p_batt_ch")
        p_batt_dis = model.addVars(H, lb=0.0, name="p_batt_dis")
        batt_soc = model.addVars(H, lb=0.0, name="batt_soc")
        p_elz = model.addVars(H, lb=0.0, name="p_elz")
        p_ro = model.addVars(H, lb=0.0, name="p_ro")
        p_ume = model.addVars(H, lb=0.0, name="p_ume")
        p_excess = model.addVars(H, lb=0.0, name="p_excess")
        gas_level = model.addVars(H, lb=0.0, name="gas_level")
        gas_supply = model.addVars(H, lb=0.0, name="gas_supply")

        # ═══════════ BINARY VARIABLES (4h blocks) ═══════════
        y_dg_blk = model.addVars(n_blocks, vtype=GRB.BINARY, name="y_dg")
        y_bch_blk = model.addVars(n_blocks, vtype=GRB.BINARY, name="y_bch")
        y_bdis_blk = model.addVars(n_blocks, vtype=GRB.BINARY, name="y_bdis")
        y_elz_blk = model.addVars(n_blocks, vtype=GRB.BINARY, name="y_elz")

        # Map hours to blocks
        y_dg = {h: y_dg_blk[h // BLOCK] for h in range(H)}
        y_bch = {h: y_bch_blk[h // BLOCK] for h in range(H)}
        y_bdis = {h: y_bdis_blk[h // BLOCK] for h in range(H)}
        y_elz = {h: y_elz_blk[h // BLOCK] for h in range(H)}

        # ═══════════ BIG-M CONSTANTS ═══════════
        M_dg = b.dg_max
        M_batt = b.batt_max  # max charge/discharge power ≈ C/1
        M_elz = b.elz_max
        inv_eff = self.p.inverter.efficiency
        bp = self.p.battery
        elzp = self.p.electrolyzer
        gsp = self.p.gas_storage
        meth = self.p.methanation

        # H2 production rate: mol H2 per kWh (from Eqs 1-2)
        h2_rate = elzp.h2_rate_mol_per_kwh
        # Paper Eq 5: M_meth = eta_meth × M_elec (direct mol conversion)
        eta_meth = meth.eta_meth  # 0.80: 1 mol H2 → 0.80 mol CH4
        # mol CH4 to m³ at STP
        mol_to_m3_stp = gsp.mol_to_m3_stp
        # m³ STP to m³ compressed (divide by compression ratio)
        comp_ratio = gsp.compression_ratio

        # Gas production: m³ STP per kWh of electrolyzer input
        gas_rate_stp_per_kwh = h2_rate * eta_meth * mol_to_m3_stp
        # Same rate in compressed m³ (for storage balance)
        gas_rate_compressed_per_kwh = gas_rate_stp_per_kwh / comp_ratio

        # Convert gas demand from m³ STP to m³ compressed
        gas_demand_compressed = self.gas_demand / comp_ratio

        # Max gas production per hour compressed (for Big-M)
        M_gas_hourly = b.elz_max * gas_rate_compressed_per_kwh + 0.1

        # RO power per hour
        ro_power_per_m3 = self.p.ro.specific_energy  # 4.38 kWh/m³

        # ═══════════ CONSTRAINTS ═══════════
        for h in range(H):
            re_h = cap_pv * self.irradiance[h] + cap_wt * self.wind[h]

            # --- Power balance (Eq 51) ---
            # Supply = Demand side
            # PV + WT + DG + Batt_dis×η_inv = E_demand + Batt_ch + ELZ + RO + Excess - UME
            model.addConstr(
                re_h + p_dg[h] + p_batt_dis[h] * inv_eff
                == self.elec_demand[h] + p_batt_ch[h] + p_elz[h]
                + p_ro[h] + p_excess[h] - p_ume[h],
                name=f"balance_{h}",
            )

            # UME bounded by demand
            model.addConstr(p_ume[h] <= self.elec_demand[h], name=f"ume_max_{h}")

            # --- DG constraints (Eqs 21-22) ---
            model.addConstr(p_dg[h] <= cap_dg, name=f"dg_cap_{h}")
            model.addConstr(p_dg[h] <= M_dg * y_dg[h], name=f"dg_on_{h}")

            # --- Battery constraints (Eqs 18-20) ---
            model.addConstr(p_batt_ch[h] <= M_batt * y_bch[h], name=f"bch_on_{h}")
            model.addConstr(p_batt_dis[h] <= M_batt * y_bdis[h], name=f"bdis_on_{h}")
            # Can't charge and discharge simultaneously
            model.addConstr(y_bch[h] + y_bdis[h] <= 1, name=f"batt_excl_{h}")

            # Battery SOC dynamics (Eq 18-19)
            if h == 0:
                model.addConstr(
                    batt_soc[h] == bp.initial_soc * cap_batt * (1.0 - bp.self_discharge_rate)
                    + p_batt_ch[h] * bp.charge_efficiency
                    - p_batt_dis[h] / bp.discharge_efficiency,
                    name=f"soc_{h}",
                )
            else:
                model.addConstr(
                    batt_soc[h] == batt_soc[h - 1] * (1.0 - bp.self_discharge_rate)
                    + p_batt_ch[h] * bp.charge_efficiency
                    - p_batt_dis[h] / bp.discharge_efficiency,
                    name=f"soc_{h}",
                )

            # SOC limits
            model.addConstr(batt_soc[h] >= bp.soc_min * cap_batt, name=f"soc_min_{h}")
            model.addConstr(batt_soc[h] <= bp.soc_max * cap_batt, name=f"soc_max_{h}")

            # --- Electrolyzer (Eqs 1-4) ---
            model.addConstr(p_elz[h] <= cap_elz, name=f"elz_cap_{h}")
            model.addConstr(p_elz[h] <= M_elz * y_elz[h], name=f"elz_on_{h}")

            # --- RO desalination (Eq 23) ---
            # RO power is fixed by water demand (must be met)
            ro_power_needed = float(self.water_demand[h] * ro_power_per_m3)
            model.addConstr(p_ro[h] == ro_power_needed, name=f"ro_power_{h}")

            # --- Gas storage balance (compressed m³, Eqs 7-12) ---
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

            # Gas storage SOC limits (compressed m³)
            model.addConstr(gas_level[h] >= gsp.soc_min * cap_gs, name=f"gs_min_{h}")
            model.addConstr(gas_level[h] <= gsp.soc_max * cap_gs, name=f"gs_max_{h}")

            # Gas supply in compressed m³ (demand converted to compressed)
            model.addConstr(gas_supply[h] <= gas_demand_compressed[h], name=f"gas_demand_{h}")

        # ═══════════ GAS DEMAND ═══════════
        # Paper requires gas for cooking. With corrected methanation (Eq 5),
        # the system should naturally produce substantial gas from excess energy.
        # Require at least 80% of annual gas demand to be met.
        total_gas_demand_comp = float(np.sum(gas_demand_compressed))
        if total_gas_demand_comp > 0:
            model.addConstr(
                gp.quicksum(gas_supply[h] for h in range(H)) >= 0.80 * total_gas_demand_comp,
                name="gas_demand_min",
            )

        # ═══════════ OBJECTIVE: min NPC (Eqs 29-36) ═══════════
        pvf = self._pvf  # present value factor for annual costs
        dr = self._dr
        n = self._n

        # --- Capital costs (Eq 30) ---
        capital = (
            self.p.pv.capital_cost * cap_pv
            + self.p.wind.capital_cost * cap_wt
            + self.p.diesel.capital_cost * cap_dg
            + self.p.battery.capital_cost * cap_batt
            + self.p.electrolyzer.capital_cost * cap_elz
            + self.p.gas_storage.capital_cost * cap_gs
            + self.p.inverter.capital_cost * (cap_pv + cap_wt)  # inverter sized to RE
        )

        # --- Replacement costs (Eq 31) ---
        rep_pv = 0.0  # PV lifetime = project lifetime, no replacement
        rep_wt = self.p.wind.replacement_cost * self._replacement_factor(self.p.wind.lifetime) * cap_wt
        # DG: 15000h lifetime. Assume ~25% capacity factor → ~2192h/yr → ~6.8yr per replacement
        dg_life_years = max(2, int(self.p.diesel.lifetime_hours / (8760 * 0.25)))
        rep_dg = self.p.diesel.replacement_cost * self._replacement_factor(dg_life_years) * cap_dg
        rep_batt = self.p.battery.replacement_cost * self._replacement_factor(self.p.battery.lifetime) * cap_batt
        rep_elz = self.p.electrolyzer.replacement_cost * self._replacement_factor(self.p.electrolyzer.lifetime) * cap_elz
        # Inverter sized proportional to max RE capacity (PV+WT)
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

        # DG O&M: $/hour × hours on (block-based)
        dg_om = self.p.diesel.om_cost * BLOCK * gp.quicksum(
            y_dg_blk[blk] for blk in range(n_blocks)
        ) * pvf

        # --- Fuel cost (Eq 36) ---
        # Annual fuel = sum(F0 × P_dg(h) + F1 × P_R × y_dg(h)) for all h
        # Fuel cost = P_fuel × annual_fuel × PV factor
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

        # --- Salvage (Eq 33-35) ---
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
            "batt_soc": batt_soc, "p_elz": p_elz, "p_ro": p_ro,
            "p_ume": p_ume, "p_excess": p_excess,
            "gas_level": gas_level, "gas_supply": gas_supply,
            "y_dg_blk": y_dg_blk, "y_elz_blk": y_elz_blk,
            "npc": npc, "capital_expr": capital, "replacement_expr": replacement,
            "om_expr": om + dg_om, "fuel_expr": fuel_cost, "salvage_expr": salvage,
            "ro_total": ro_total,
            "annual_fuel_expr": annual_fuel_expr,
            "gas_rate_stp_per_kwh": gas_rate_stp_per_kwh,
            "BLOCK": BLOCK, "n_blocks": n_blocks,
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
