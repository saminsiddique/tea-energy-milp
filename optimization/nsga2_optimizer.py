"""NSGA-II wrapper (pymoo) for the rule-based EMS simulator.

Single-objective: minimise NPC.
Constraint: LPSP <= params.nsga2.lpsp_max (1% default).

Paper Table 5 parameters drive the algorithm configuration:
  - Population size: 500
  - Generations: 500
  - SBX crossover rate: 0.9
  - Polynomial mutation rate: 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from config.nsga_parameters import NSGASystemParams, NSGACapacityBounds
from optimization.ems_simulator import simulate, SimulationResult


# ============================================================
# Population record (single individual in the final population)
# ============================================================

@dataclass
class PopulationRecord:
    pv_kw: float
    wind_kw: float
    dg_kw: float
    batt_kwh: float
    elz_kw: float
    gas_storage_m3: float
    npc: float
    lpsp: float           # actual LPSP fraction (not violation); = g1 + lpsp_max
    gas_violation: float  # g2 value; <= 0 means constraint satisfied
    feasible: bool


# ============================================================
# Problem definition
# ============================================================

class HESProblem(ElementwiseProblem):
    """Sizing problem over (PV, Wind, DG, Batt, ELZ, GS).

    Objective: minimise NPC ($).
    Constraints:
      g1: LPSP - lpsp_max <= 0                      (reliability)
      g2: min_gas_coverage * gas_demand - ch4_del <= 0  (gas chain usage)

    The gas coverage constraint is needed because, without it, NSGA-II
    correctly identifies the electrolyzer and gas storage as pure cost
    (gas demand is not otherwise enforced) and drives them to zero. The
    paper's Table 6 implies ~10 % gas coverage (ELZ = 139 kW produces
    ~11,663 m3/yr CH4 against 120,450 m3/yr demand), so we mirror that
    by requiring at least `min_gas_coverage` of annual gas demand.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        params: NSGASystemParams,
        bounds: NSGACapacityBounds,
        lpsp_max: float,
        min_gas_coverage: float = 0.20,  # paper sizes produce ~21% coverage with our weather data
    ):
        xl = np.array([
            bounds.pv_min, bounds.wind_min, bounds.dg_min,
            bounds.batt_min, bounds.elz_min, bounds.gas_storage_min,
        ])
        xu = np.array([
            bounds.pv_max, bounds.wind_max, bounds.dg_max,
            bounds.batt_max, bounds.elz_max, bounds.gas_storage_max,
        ])
        super().__init__(n_var=6, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)
        self.data = data
        self.params = params
        self.lpsp_max = lpsp_max
        self.min_gas_coverage = min_gas_coverage
        self._total_gas_demand = float(np.sum(data["gas_demand"]))

    def _evaluate(self, x, out, *args, **kwargs):
        sizes = {
            "pv_kw": float(x[0]),
            "wind_kw": float(x[1]),
            "dg_kw": float(x[2]),
            "batt_kwh": float(x[3]),
            "elz_kw": float(x[4]),
            "gas_storage_m3": float(x[5]),
        }
        res = simulate(sizes, self.data, self.params)
        gas_required = self.min_gas_coverage * self._total_gas_demand
        out["F"] = [res.npc]
        out["G"] = [
            res.lpsp - self.lpsp_max,
            # Constrain DELIVERY (gas actually drawn from storage) so the
            # optimizer must size both electrolyzer AND gas storage.
            gas_required - res.annual_ch4_delivered_stp,
        ]


# ============================================================
# Runner
# ============================================================

@dataclass
class NSGA2RunSummary:
    best: SimulationResult
    n_evals: int
    history: list  # list of (gen, best_npc, best_lpsp)
    final_population: list  # list[PopulationRecord], sorted: feasible first, then NPC asc


def run_nsga2(
    data: Dict[str, np.ndarray],
    params: NSGASystemParams,
    bounds: NSGACapacityBounds | None = None,
    verbose: bool = True,
) -> NSGA2RunSummary:
    """Run the paper's NSGA-II and return the best feasible solution."""
    bounds = bounds or params.bounds
    nsga2_cfg = params.nsga2

    problem = HESProblem(data, params, bounds, nsga2_cfg.lpsp_max)

    # Paper uses NSGA-II but the inner problem is single-objective.
    # pymoo's NSGA2 class handles both; we use it directly so the Table 5
    # parameters (SBX, PM, pop, gens) map 1:1 onto the paper's algorithm.
    algorithm = NSGA2(
        pop_size=nsga2_cfg.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=nsga2_cfg.p_crossover, eta=nsga2_cfg.sbx_eta),
        mutation=PM(prob=nsga2_cfg.p_mutation, eta=nsga2_cfg.pm_eta),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", nsga2_cfg.n_gens),
        seed=nsga2_cfg.seed,
        verbose=verbose,
        save_history=False,
    )

    # Extract full final population
    pop_X = res.pop.get("X")           # (n_pop, 6)
    pop_F = res.pop.get("F")           # (n_pop, 1)
    pop_G = res.pop.get("G")           # (n_pop, 2)
    pop_feas = res.pop.get("feasible").flatten()  # (n_pop,) bool
    lpsp_max = nsga2_cfg.lpsp_max

    final_pop: list[PopulationRecord] = []
    for i in range(len(pop_X)):
        final_pop.append(PopulationRecord(
            pv_kw=float(pop_X[i, 0]),
            wind_kw=float(pop_X[i, 1]),
            dg_kw=float(pop_X[i, 2]),
            batt_kwh=float(pop_X[i, 3]),
            elz_kw=float(pop_X[i, 4]),
            gas_storage_m3=float(pop_X[i, 5]),
            npc=float(pop_F[i, 0]),
            lpsp=float(pop_G[i, 0]) + lpsp_max,  # g1 = lpsp - lpsp_max => lpsp = g1 + lpsp_max
            gas_violation=float(pop_G[i, 1]),
            feasible=bool(pop_feas[i]),
        ))
    final_pop.sort(key=lambda r: (not r.feasible, r.npc))

    # Extract best feasible solution
    if res.X is None:
        raise RuntimeError("NSGA-II returned no feasible solution")

    # pymoo single-objective with constraints: res.X is the best feasible x
    # (if any feasible was found). Shape may be (n_var,) or (pop, n_var).
    x_best = res.X
    if x_best.ndim > 1:
        # Find the feasible solution with lowest F
        f_vals = res.F
        g_vals = res.G
        f_2d = np.atleast_2d(f_vals)
        if g_vals is not None:
            g_2d = np.atleast_2d(g_vals)
            feas_mask = np.all(g_2d <= 0, axis=1)
            if feas_mask.any():
                feas_idx = np.where(feas_mask)[0]
                best_idx = feas_idx[np.argmin(f_2d[feas_idx, 0])]
            else:
                best_idx = int(np.argmin(f_2d[:, 0]))
        else:
            best_idx = int(np.argmin(f_2d[:, 0]))
        x_best = x_best[best_idx]

    best_sizes = {
        "pv_kw": float(x_best[0]),
        "wind_kw": float(x_best[1]),
        "dg_kw": float(x_best[2]),
        "batt_kwh": float(x_best[3]),
        "elz_kw": float(x_best[4]),
        "gas_storage_m3": float(x_best[5]),
    }
    best_result = simulate(best_sizes, data, params)

    return NSGA2RunSummary(
        best=best_result,
        n_evals=res.algorithm.evaluator.n_eval,
        history=[],
        final_population=final_pop,
    )
