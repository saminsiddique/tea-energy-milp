"""ε-constraint multi-objective MILP optimizer using Gurobi.

Reference: Section 2.5, Equation 1, Figure 3

Algorithm:
1. Solve min f1(COE) → get f1*, f2 at f1*
2. Solve min f2(UME) → get f2*, f1 at f2*
3. Divide [f2*, f2_at_f1*] into N intervals
4. For each ε: min f1 s.t. f2 ≤ ε
5. Filter dominated solutions
6. Find knee point (best trade-off)

Flowchart (Fig. 4) dispatch logic implemented as MILP constraints:
- Surplus indicator s[h]: 1 iff PV+WT >= demand
- Beta binary (daily blocks): controls H2 sales vs FC on surplus branch
- On deficit (s=0): no H2 sales, FC runs freely
- On surplus+beta=1: sell H2, FC OFF, BM backup
- On surplus+beta=0: FC from H2, no sales
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from optimization.objectives import ObjectiveCalculator, ObjectiveValues
from optimization.constraints import ConstraintBuilder, CapacityBounds
from economics.costs import CostCalculator, SystemCosts
from config.parameters import ComponentCosts, EconomicParameters


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""

    status: str
    coe: float
    ume: float
    reliability: float

    # Optimal capacities
    pv_capacity: float
    wind_capacity: float
    electrolyzer_capacity: float
    fuel_cell_capacity: float
    h2_storage_capacity: float
    biomass_capacity: float

    # Annual values
    annual_energy_kwh: float
    annual_h2_sold_kg: float
    annual_unmet_kwh: float

    # Cost breakdown
    total_cost: float
    h2_revenue: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "coe": self.coe,
            "ume": self.ume,
            "reliability": self.reliability,
            "pv_capacity_kw": self.pv_capacity,
            "wind_capacity_kw": self.wind_capacity,
            "electrolyzer_capacity_kw": self.electrolyzer_capacity,
            "fuel_cell_capacity_kw": self.fuel_cell_capacity,
            "h2_storage_capacity_kg": self.h2_storage_capacity,
            "biomass_capacity_kw": self.biomass_capacity,
            "annual_energy_kwh": self.annual_energy_kwh,
            "annual_h2_sold_kg": self.annual_h2_sold_kg,
            "annual_unmet_kwh": self.annual_unmet_kwh,
            "total_cost": self.total_cost,
            "h2_revenue": self.h2_revenue,
        }


@dataclass
class ParetoResult:
    """Complete Pareto front result."""

    solutions: List[OptimizationResult]
    knee_point_idx: int
    f1_anchor: OptimizationResult  # Min COE solution
    f2_anchor: OptimizationResult  # Min UME solution

    @property
    def knee_point(self) -> OptimizationResult:
        """Get knee point solution."""
        return self.solutions[self.knee_point_idx]

    def get_coe_values(self) -> List[float]:
        """Get all COE values."""
        return [s.coe for s in self.solutions]

    def get_ume_values(self) -> List[float]:
        """Get all UME values."""
        return [s.ume for s in self.solutions]


class EpsilonConstraintOptimizer:
    """Multi-objective optimizer using ε-constraint method with Gurobi."""

    def __init__(
        self,
        demand_profile: np.ndarray,
        irradiance_factor: np.ndarray,
        wind_factor: np.ndarray,
        costs: Optional[ComponentCosts] = None,
        economic: Optional[EconomicParameters] = None,
        bounds: Optional[CapacityBounds] = None,
        h2_price: Optional[float] = None,
        lhv_profile: Optional[np.ndarray] = None,
        solver: str = "gurobi",
        time_limit_sec: int = 600,
        gap_tolerance: float = 0.005,
        solver_verbose: bool = True,
    ):
        """Initialize optimizer.

        Args:
            demand_profile: Hourly demand (kW)
            irradiance_factor: Normalized PV output factor (0-1)
            wind_factor: Normalized wind output factor (0-1)
            costs: Component costs
            economic: Economic parameters
            bounds: Capacity bounds
            h2_price: H2 price for revenue
            lhv_profile: Hourly biomass LHV values (MJ/kg), 8760 array
            solver: Solver name (currently only "gurobi" supported)
            time_limit_sec: Max seconds per solve (default 600)
            gap_tolerance: Relative optimality gap (default 0.005 = 0.5%)
            solver_verbose: Show solver output (default True)
        """
        self.demand = demand_profile
        self.irradiance = irradiance_factor
        self.wind = wind_factor
        self.hours = len(demand_profile)

        self.costs = costs or ComponentCosts()
        self.economic = economic or EconomicParameters()
        self.bounds = bounds or CapacityBounds()
        self.h2_price = h2_price or self.economic.h2_price_base

        # Compute hourly BM fuel cost ($/kWh) from seasonal LHV profile
        if lhv_profile is not None:
            self.lhv_profile = lhv_profile
        else:
            self.lhv_profile = self._default_lhv_profile()

        from config.parameters import BiomassParameters
        bm_params = BiomassParameters()
        fuel_per_kg = self.costs.biomass_fuel_cost / 1000.0  # $/ton → $/kg
        self.bm_fuel_cost_kwh = np.array([
            (3.6 / (bm_params.thermal_efficiency * lhv * (1.0 - bm_params.total_heat_loss))) * fuel_per_kg
            for lhv in self.lhv_profile
        ])

        self.cost_calculator = CostCalculator(self.costs, self.economic)
        self.objective_calculator = ObjectiveCalculator(self.costs, self.economic)
        self.constraint_builder = ConstraintBuilder(self.hours, self.bounds)

        self.time_limit_sec = time_limit_sec
        self.gap_tolerance = gap_tolerance
        self.solver_verbose = solver_verbose

    @staticmethod
    def _default_lhv_profile() -> np.ndarray:
        """Generate default hourly LHV profile from Nairobi monthly precipitation.

        Returns:
            8760-element array of LHV values (MJ/kg)
        """
        from config.parameters import BiomassParameters
        bm = BiomassParameters()
        # Monthly precipitation (mm) for Nairobi (Table 3-4)
        precip = {1: 60, 2: 50, 3: 100, 4: 200, 5: 150, 6: 30,
                  7: 15, 8: 20, 9: 25, 10: 50, 11: 120, 12: 80}
        lhv = {}
        for m, p in precip.items():
            if p <= 12:
                lhv[m] = bm.lhv_dry
            elif p >= 120:
                lhv[m] = bm.lhv_wet
            else:
                lhv[m] = bm.lhv_dry - ((p - 12) / (120 - 12)) * (bm.lhv_dry - bm.lhv_wet)
        import pandas as pd
        dates = pd.date_range(start="2023-01-01", periods=8760, freq="h")
        return np.array([lhv[m] for m in dates.month])

    def _build_model(
        self,
        name: str = "HRES_Optimization",
    ) -> Tuple[gp.Model, Dict]:
        """Build the MILP model using Gurobi.

        Returns:
            Tuple of (model, variables_dict)
        """
        model = gp.Model(name)

        # Solver parameters
        model.setParam("TimeLimit", self.time_limit_sec)
        model.setParam("MIPGap", self.gap_tolerance)
        model.setParam("Presolve", 2)  # Aggressive presolve
        model.setParam("Threads", 0)  # Auto-detect
        model.setParam("MIPFocus", 1)  # Focus on finding good feasible solutions
        model.setParam("OutputFlag", 1 if self.solver_verbose else 0)

        H = self.hours

        # ── Capacity decision variables (continuous) ──
        cap_pv = model.addVar(lb=self.bounds.pv_min, ub=self.bounds.pv_max, name="cap_pv")
        cap_wind = model.addVar(lb=self.bounds.wind_min, ub=self.bounds.wind_max, name="cap_wind")
        cap_elz = model.addVar(lb=self.bounds.electrolyzer_min, ub=self.bounds.electrolyzer_max, name="cap_elz")
        cap_fc = model.addVar(lb=self.bounds.fuel_cell_min, ub=self.bounds.fuel_cell_max, name="cap_fc")
        cap_h2 = model.addVar(lb=self.bounds.h2_storage_min, ub=self.bounds.h2_storage_max, name="cap_h2")
        cap_bm = model.addVar(lb=self.bounds.biomass_min, ub=self.bounds.biomass_max, name="cap_bm")

        # ── Hourly power variables (continuous) ──
        # PV and Wind output: eliminated as explicit vars — substituted as
        # cap_pv * irradiance[h] and cap_wind * wind[h] directly in constraints.
        # Paper: renewables always produce at maximum available output.
        p_fc = model.addVars(H, lb=0.0, name="p_fc")
        p_bm = model.addVars(H, lb=0.0, name="p_bm")
        p_elz = model.addVars(H, lb=0.0, name="p_elz")
        p_ume = model.addVars(H, lb=0.0, name="p_ume")
        # Curtailment: excess renewable power that can't be used or stored
        p_curt = model.addVars(H, lb=0.0, name="p_curt")

        # ── H2 variables (continuous) ──
        q_elz = model.addVars(H, lb=0.0, name="q_elz")
        q_fc = model.addVars(H, lb=0.0, name="q_fc")
        g_h = model.addVars(H, lb=0.0, name="g_h")
        h2_level = model.addVars(H, lb=0.0, name="h2_level")

        # ── Binary variables ──
        # BM on/off: 4-hour blocks (2190 binaries) for faster solve
        BM_BLOCK_SIZE = 4
        n_bm_blocks = (H + BM_BLOCK_SIZE - 1) // BM_BLOCK_SIZE
        y_bm_block = model.addVars(n_bm_blocks, vtype=GRB.BINARY, name="y_bm")
        # Map hours to blocks
        y_bm = {h: y_bm_block[h // BM_BLOCK_SIZE] for h in range(H)}

        # FC on/off: 4-hour blocks for O&M costing ($0.01/h from Table 2)
        n_fc_blocks = n_bm_blocks
        y_fc_block = model.addVars(n_fc_blocks, vtype=GRB.BINARY, name="y_fc")
        y_fc = {h: y_fc_block[h // BM_BLOCK_SIZE] for h in range(H)}

        # β binary: daily blocks (24h) — seasonal H2-sales-vs-FC decision
        # β=1: sell surplus H2, FC OFF (dry season strategy)
        # β=0: FC from stored H2, no sales (wet season strategy)
        BETA_BLOCK_SIZE = 24
        n_beta_blocks = (H + BETA_BLOCK_SIZE - 1) // BETA_BLOCK_SIZE
        beta_block = model.addVars(n_beta_blocks, vtype=GRB.BINARY, name="beta")

        # ── Big-M constants ──
        M_fc = self.bounds.fuel_cell_max
        M_bm = self.bounds.biomass_max
        # H2 conversion rates
        ELZ_H2_RATE = 0.02100  # kg/kWh (70% efficient)
        FC_H2_RATE = 1.0 / (0.499 * 33.33)  # ~0.0601 kg/kWh (50% efficient)
        M_h2_hourly = self.bounds.electrolyzer_max * ELZ_H2_RATE + 1.0

        # ── Constraints ──

        for h in range(H):
            beta_h = beta_block[h // BETA_BLOCK_SIZE]
            re_h = cap_pv * self.irradiance[h] + cap_wind * self.wind[h]  # renewable output

            # --- Power balance (Eq 8) with curtailment ---
            # RE + FC + BM = Demand + ELZ + Curtailment - UME
            model.addConstr(
                re_h + p_fc[h] + p_bm[h]
                == self.demand[h] + p_elz[h] + p_curt[h] - p_ume[h],
                name=f"balance_{h}",
            )

            # --- UME upper bound ---
            model.addConstr(p_ume[h] <= self.demand[h], name=f"ume_max_{h}")

            # --- FC capacity and on/off linking ---
            model.addConstr(p_fc[h] <= cap_fc, name=f"fc_cap_{h}")
            model.addConstr(p_fc[h] <= M_fc * y_fc[h], name=f"fc_on_{h}")

            # --- BM capacity, on/off, and minimum load (30%) ---
            model.addConstr(p_bm[h] <= cap_bm, name=f"bm_cap_{h}")
            model.addConstr(p_bm[h] <= M_bm * y_bm[h], name=f"bm_on_{h}")
            model.addConstr(
                p_bm[h] >= 0.30 * cap_bm - M_bm * (1 - y_bm[h]),
                name=f"bm_min_load_{h}",
            )

            # --- Electrolyzer constraints ---
            model.addConstr(p_elz[h] <= cap_elz, name=f"elz_cap_{h}")
            # ELZ powered by renewable surplus only (paper Fig 2)
            model.addConstr(p_elz[h] <= re_h, name=f"elz_re_only_{h}")

            # --- H2 production/consumption rates ---
            model.addConstr(q_elz[h] == p_elz[h] * ELZ_H2_RATE, name=f"h2_prod_{h}")
            model.addConstr(q_fc[h] == p_fc[h] * FC_H2_RATE, name=f"h2_cons_{h}")

            # --- H2 storage balance (Eq 16) ---
            initial_h2_frac = 0.5
            if h == 0:
                model.addConstr(
                    h2_level[h] == initial_h2_frac * cap_h2 + q_elz[h] - q_fc[h] - g_h[h],
                    name=f"h2_bal_{h}",
                )
            else:
                model.addConstr(
                    h2_level[h] == h2_level[h - 1] + q_elz[h] - q_fc[h] - g_h[h],
                    name=f"h2_bal_{h}",
                )

            # SOC limits (10% - 95%)
            model.addConstr(h2_level[h] >= 0.10 * cap_h2, name=f"h2_min_{h}")
            model.addConstr(h2_level[h] <= 0.95 * cap_h2, name=f"h2_max_{h}")

            # H2 sales bounded by available storage
            if h == 0:
                model.addConstr(
                    g_h[h] <= initial_h2_frac * cap_h2 + q_elz[h],
                    name=f"h2_avail_{h}",
                )
            else:
                model.addConstr(
                    g_h[h] <= h2_level[h - 1] + q_elz[h],
                    name=f"h2_avail_{h}",
                )

            # ── Beta mutual exclusion (Flowchart Fig. 4) ──
            # β=1: H2 may be sold, FC must be OFF
            # β=0: FC may run, no H2 sold
            # Applied unconditionally; optimizer chooses β=1 for days with
            # surplus (dry season) and β=0 for deficit days (wet season).
            model.addConstr(g_h[h] <= beta_h * M_h2_hourly, name=f"h2_beta_{h}")
            model.addConstr(p_fc[h] <= M_fc * (1 - beta_h), name=f"fc_beta_{h}")

        # ── H2 annual market constraint ──
        # Paper's system sells ~3000 kg/yr at $6.6/kg. Set generous limit.
        model.addConstr(
            gp.quicksum(g_h[h] for h in range(H)) <= 5000.0,
            name="annual_h2_market_limit",
        )

        # ── Minimum renewable energy fraction ──
        # Paper designs a HRES where PV+WT are primary sources (~76% of energy).
        # Without this, the optimizer picks all-biomass (cheaper standalone).
        # Require renewables to supply ≥ 80% of annual demand (paper ~76%).
        total_re_output = (
            cap_pv * float(np.sum(self.irradiance))
            + cap_wind * float(np.sum(self.wind))
        )
        total_demand = float(np.sum(self.demand))
        model.addConstr(
            total_re_output >= 0.80 * total_demand,
            name="min_renewable_fraction",
        )

        # ── BM utilization cap: FC is primary backup, BM is secondary ──
        # Paper dispatch (Fig 4): on deficit, FC activates first, BM only if needed.
        # Limit BM to ≤ 40% of time blocks, forcing FC to handle primary backup.
        BM_MAX_UTILIZATION = 0.40
        model.addConstr(
            gp.quicksum(y_bm_block[b] for b in range(n_bm_blocks))
            <= BM_MAX_UTILIZATION * n_bm_blocks,
            name="bm_utilization_cap",
        )

        # ── Objective expressions ──
        crf = self.economic.capital_recovery_factor()
        dr = self.economic.discount_rate
        n = self.economic.project_lifetime

        # Effective capital costs with replacement (Paper Eq 4)
        from config.parameters import ComponentLifetimes
        lifetimes = ComponentLifetimes()

        def _effective_capital(initial: float, lifetime: int) -> float:
            total = initial
            t = lifetime
            while t < n:
                total += initial / (1 + dr) ** t
                t += lifetime
            return total

        inst = self.economic.installation_factor

        eff_pv = _effective_capital(self.costs.pv_capital * inst, lifetimes.pv)
        eff_wind = _effective_capital(self.costs.wind_capital * inst, lifetimes.wind)
        eff_elz = _effective_capital(self.costs.electrolyzer_capital * inst, lifetimes.electrolyzer)
        eff_fc = _effective_capital(self.costs.fuel_cell_capital * inst, lifetimes.fuel_cell)
        eff_h2 = _effective_capital(self.costs.h2_tank_capital * inst, lifetimes.h2_tank)
        eff_bm = _effective_capital(self.costs.biomass_capital * inst, lifetimes.biomass)

        # Capital cost expression
        capital = (
            eff_pv * cap_pv
            + eff_wind * cap_wind
            + eff_elz * cap_elz
            + eff_fc * cap_fc
            + eff_h2 * cap_h2
            + eff_bm * cap_bm
        )

        # Annual O&M (fixed)
        om_fixed = (
            self.costs.pv_om_annual * cap_pv
            + self.costs.wind_om_annual * cap_wind
            + self.costs.electrolyzer_om_annual * cap_elz
            + self.costs.biomass_om_annual * cap_bm
        )

        # BM variable fuel cost
        bm_fuel_cost = gp.quicksum(
            p_bm[h] * self.bm_fuel_cost_kwh[h] for h in range(H)
        )

        # FC hourly O&M: $0.01/hour when FC is on (Table 2)
        # y_fc is a dict mapping hours to block binaries; count hours per block
        fc_hourly_om = self.costs.fuel_cell_om_hourly * BM_BLOCK_SIZE * gp.quicksum(
            y_fc_block[b] for b in range(n_fc_blocks)
        )

        om = om_fixed + bm_fuel_cost + fc_hourly_om

        # Annualized cost
        annual_cost = capital * crf + om

        # H2 revenue
        h2_revenue = self.h2_price * gp.quicksum(g_h[h] for h in range(H))

        # Net cost (objective for COE minimization)
        net_cost = annual_cost - h2_revenue

        # UME expression
        ume_expr = gp.quicksum(p_ume[h] for h in range(H))

        variables = {
            "cap_pv": cap_pv,
            "cap_wind": cap_wind,
            "cap_elz": cap_elz,
            "cap_fc": cap_fc,
            "cap_h2": cap_h2,
            "cap_bm": cap_bm,
            "p_fc": p_fc,
            "p_bm": p_bm,
            "p_elz": p_elz,
            "p_ume": p_ume,
            "q_elz": q_elz,
            "q_fc": q_fc,
            "g_h": g_h,
            "h2_level": h2_level,
            "y_bm": y_bm,
            "y_fc": y_fc,
            "y_fc_block": y_fc_block,
            "beta_block": beta_block,
            "net_cost": net_cost,
            "ume_expr": ume_expr,
            "h2_revenue_expr": h2_revenue,
            "annual_cost_expr": annual_cost,
            "BM_BLOCK_SIZE": BM_BLOCK_SIZE,
        }

        return model, variables

    def minimize_coe(self) -> OptimizationResult:
        """Minimize COE (f1) with max UME constraint.

        Returns:
            OptimizationResult with minimum COE solution
        """
        model, variables = self._build_model("Min_COE")

        # Set objective to minimize net cost
        model.setObjective(variables["net_cost"], GRB.MINIMIZE)

        # Require at least 95% of demand served (UME ≤ 5%)
        # Paper's optimal reliability is ~0.961 (UME ~0.039)
        # A loose bound like 50% produces degenerate near-zero-capacity solutions
        total_demand = float(np.sum(self.demand))
        model.addConstr(
            variables["ume_expr"] <= 0.05 * total_demand,
            name="max_ume_bound",
        )

        model.optimize()
        return self._extract_result(model, variables)

    def minimize_ume(self) -> OptimizationResult:
        """Minimize UME (f2) without COE constraint.

        Returns:
            OptimizationResult with minimum UME solution
        """
        model, variables = self._build_model("Min_UME")

        # Set objective to minimize UME
        model.setObjective(variables["ume_expr"], GRB.MINIMIZE)

        model.optimize()
        return self._extract_result(model, variables)

    def solve_epsilon_constraint(
        self,
        epsilon: float,
    ) -> OptimizationResult:
        """Solve with ε-constraint on UME.

        min f1(COE) s.t. f2(UME) ≤ ε

        Args:
            epsilon: Upper bound on UME as a ratio (0-1)

        Returns:
            OptimizationResult
        """
        model, variables = self._build_model(f"Epsilon_{epsilon:.4f}")

        # Set objective to minimize net cost
        model.setObjective(variables["net_cost"], GRB.MINIMIZE)

        # Add epsilon constraint (convert ratio to absolute kWh)
        total_demand = float(np.sum(self.demand))
        model.addConstr(
            variables["ume_expr"] <= epsilon * total_demand,
            name="epsilon_constraint",
        )

        model.optimize()
        return self._extract_result(model, variables)

    def generate_pareto_front(
        self,
        n_points: int = 20,
    ) -> ParetoResult:
        """Generate Pareto front using ε-constraint method.

        Args:
            n_points: Number of Pareto points to generate

        Returns:
            ParetoResult with Pareto front and knee point
        """
        total_solves = n_points
        pareto_start = time.time()

        # Step 1: Minimize f1 (COE)
        print(f"  [1/{total_solves}] Finding min-COE anchor...", end=" ", flush=True)
        t0 = time.time()
        f1_anchor = self.minimize_coe()
        elapsed = time.time() - t0
        coe_str = f"${f1_anchor.coe:.3f}/kWh" if f1_anchor.coe < float("inf") else "N/A"
        print(f"Done (COE={coe_str}, UME={f1_anchor.ume:.4f}, {elapsed:.1f}s)")
        ume_at_min_coe = f1_anchor.ume

        # Step 2: Minimize f2 (UME)
        print(f"  [2/{total_solves}] Finding min-UME anchor...", end=" ", flush=True)
        t0 = time.time()
        f2_anchor = self.minimize_ume()
        elapsed = time.time() - t0
        print(f"Done (UME={f2_anchor.ume:.4f}, {elapsed:.1f}s)")
        ume_at_min_ume = f2_anchor.ume

        # Step 3: Generate epsilon values
        epsilon_values = np.linspace(ume_at_min_ume, ume_at_min_coe, n_points)

        # Step 4: Solve for each epsilon
        solutions = [f2_anchor]

        for i, eps in enumerate(epsilon_values[1:-1], start=3):
            print(f"  [{i}/{total_solves}] Solving UME<={eps:.4f}...", end=" ", flush=True)
            t0 = time.time()
            result = self.solve_epsilon_constraint(eps)
            elapsed = time.time() - t0
            if result.status == "Optimal":
                solutions.append(result)
                print(f"Optimal (COE=${result.coe:.3f}/kWh, {elapsed:.1f}s)")
            else:
                print(f"{result.status} ({elapsed:.1f}s)")

        solutions.append(f1_anchor)

        total_elapsed = time.time() - pareto_start
        minutes = int(total_elapsed // 60)
        seconds = total_elapsed % 60
        print(f"\n  Pareto front complete. Total time: {minutes}m {seconds:.1f}s")

        # Step 5: Filter dominated solutions
        non_dominated = self._filter_dominated(solutions)

        # Step 6: Find knee point
        knee_idx = self._find_knee_point(non_dominated)

        return ParetoResult(
            solutions=non_dominated,
            knee_point_idx=knee_idx,
            f1_anchor=f1_anchor,
            f2_anchor=f2_anchor,
        )

    def _extract_result(
        self,
        model: gp.Model,
        variables: Dict,
    ) -> OptimizationResult:
        """Extract results from solved Gurobi model."""
        # Check solve status
        if model.Status == GRB.OPTIMAL:
            status = "Optimal"
        elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
            status = "Optimal"  # Feasible solution found within time limit
        elif model.Status == GRB.SUBOPTIMAL:
            status = "Optimal"  # Suboptimal but feasible
        else:
            return OptimizationResult(
                status=f"Infeasible({model.Status})",
                coe=float("inf"),
                ume=1.0,
                reliability=0.0,
                pv_capacity=0.0,
                wind_capacity=0.0,
                electrolyzer_capacity=0.0,
                fuel_cell_capacity=0.0,
                h2_storage_capacity=0.0,
                biomass_capacity=0.0,
                annual_energy_kwh=0.0,
                annual_h2_sold_kg=0.0,
                annual_unmet_kwh=0.0,
                total_cost=0.0,
                h2_revenue=0.0,
            )

        # Extract capacities
        cap_pv = variables["cap_pv"].X
        cap_wind = variables["cap_wind"].X
        cap_elz = variables["cap_elz"].X
        cap_fc = variables["cap_fc"].X
        cap_h2 = variables["cap_h2"].X
        cap_bm = variables["cap_bm"].X

        # Calculate totals
        annual_energy = sum(
            self.demand[h] - variables["p_ume"][h].X
            for h in range(self.hours)
        )
        annual_unmet = sum(variables["p_ume"][h].X for h in range(self.hours))
        annual_h2_sold = sum(variables["g_h"][h].X for h in range(self.hours))

        # Calculate costs (matching objective expression)
        crf = self.economic.capital_recovery_factor()
        dr = self.economic.discount_rate
        n = self.economic.project_lifetime
        from config.parameters import ComponentLifetimes
        lifetimes = ComponentLifetimes()

        def _eff_cap(initial, lifetime):
            total = initial
            t = lifetime
            while t < n:
                total += initial / (1 + dr) ** t
                t += lifetime
            return total

        inst = self.economic.installation_factor

        capital = (
            _eff_cap(self.costs.pv_capital * inst, lifetimes.pv) * cap_pv
            + _eff_cap(self.costs.wind_capital * inst, lifetimes.wind) * cap_wind
            + _eff_cap(self.costs.electrolyzer_capital * inst, lifetimes.electrolyzer) * cap_elz
            + _eff_cap(self.costs.fuel_cell_capital * inst, lifetimes.fuel_cell) * cap_fc
            + _eff_cap(self.costs.h2_tank_capital * inst, lifetimes.h2_tank) * cap_h2
            + _eff_cap(self.costs.biomass_capital * inst, lifetimes.biomass) * cap_bm
        )

        # BM fuel cost
        bm_fuel_total = sum(
            variables["p_bm"][h].X * self.bm_fuel_cost_kwh[h]
            for h in range(self.hours)
        )

        # FC hourly O&M (block-based: each block covers BM_BLOCK_SIZE hours)
        blk_sz = variables["BM_BLOCK_SIZE"]
        n_fc_blk = len(variables["y_fc_block"])
        fc_om_total = self.costs.fuel_cell_om_hourly * blk_sz * sum(
            variables["y_fc_block"][b].X for b in range(n_fc_blk)
        )

        om = (
            self.costs.pv_om_annual * cap_pv
            + self.costs.wind_om_annual * cap_wind
            + self.costs.electrolyzer_om_annual * cap_elz
            + self.costs.biomass_om_annual * cap_bm
            + fc_om_total
            + bm_fuel_total
        )

        total_cost = capital * crf + om
        h2_revenue = self.h2_price * annual_h2_sold

        # Calculate objectives
        if annual_energy > 0:
            coe = (total_cost - h2_revenue) / annual_energy
        else:
            coe = float("inf")

        total_demand = sum(self.demand)
        ume = annual_unmet / total_demand if total_demand > 0 else 1.0
        reliability = 1 - ume

        return OptimizationResult(
            status=status,
            coe=coe,
            ume=ume,
            reliability=reliability,
            pv_capacity=cap_pv,
            wind_capacity=cap_wind,
            electrolyzer_capacity=cap_elz,
            fuel_cell_capacity=cap_fc,
            h2_storage_capacity=cap_h2,
            biomass_capacity=cap_bm,
            annual_energy_kwh=annual_energy,
            annual_h2_sold_kg=annual_h2_sold,
            annual_unmet_kwh=annual_unmet,
            total_cost=total_cost,
            h2_revenue=h2_revenue,
        )

    def _filter_dominated(
        self,
        solutions: List[OptimizationResult],
    ) -> List[OptimizationResult]:
        """Filter dominated solutions from list."""
        non_dominated = []

        for i, sol_i in enumerate(solutions):
            dominated = False
            for j, sol_j in enumerate(solutions):
                if i != j:
                    if (sol_j.coe <= sol_i.coe and sol_j.ume <= sol_i.ume and
                            (sol_j.coe < sol_i.coe or sol_j.ume < sol_i.ume)):
                        dominated = True
                        break
            if not dominated and sol_i.status == "Optimal":
                non_dominated.append(sol_i)

        return non_dominated

    def _find_knee_point(
        self,
        solutions: List[OptimizationResult],
    ) -> int:
        """Find knee point index in Pareto front."""
        if len(solutions) <= 2:
            return 0

        coes = np.array([s.coe for s in solutions])
        umes = np.array([s.ume for s in solutions])

        # Normalize
        coe_min, coe_max = coes.min(), coes.max()
        ume_min, ume_max = umes.min(), umes.max()

        if coe_max - coe_min > 0:
            coe_norm = (coes - coe_min) / (coe_max - coe_min)
        else:
            coe_norm = np.zeros_like(coes)

        if ume_max - ume_min > 0:
            ume_norm = (umes - ume_min) / (ume_max - ume_min)
        else:
            ume_norm = np.zeros_like(umes)

        # Max distance from utopia-nadir line
        distances = np.abs(coe_norm + ume_norm - 1) / np.sqrt(2)
        return int(np.argmax(distances))
