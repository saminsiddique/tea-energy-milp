"""Technical constraints for MILP optimization.

Reference: Section 2.5.2, Equations 8-10, 32

Constraints include:
- Power balance (Eq 8)
- Dispatch mode selection (Eq 9-10)
- Component capacity limits
- Hydrogen storage balance (Eq 16)
- Feedstock availability (Eq 32)
- Minimum operating loads
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp

from config.parameters import (
    PVParameters,
    WindParameters,
    ElectrolyzerParameters,
    FuelCellParameters,
    HydrogenStorageParameters,
    BiomassParameters,
)


@dataclass
class CapacityBounds:
    """Bounds for component capacities."""

    pv_min: float = 0.0
    pv_max: float = 150.0      # ~3x paper optimal (41.8 kW)
    wind_min: float = 10.0      # HRES design: must include wind
    wind_max: float = 150.0    # ~5x paper optimal (30.1 kW)
    electrolyzer_min: float = 20.0  # HRES design: meaningful H2 production
    electrolyzer_max: float = 150.0  # ~3x paper optimal (40.3 kW)
    fuel_cell_min: float = 12.0     # HRES design: FC is primary backup
    fuel_cell_max: float = 60.0      # ~4x paper optimal (15.1 kW)
    h2_storage_min: float = 50.0     # HRES design: meaningful H2 buffer for FC
    h2_storage_max: float = 500.0    # ~5x paper optimal (~100 kg)
    biomass_min: float = 0.0
    biomass_max: float = 100.0  # ~3x paper optimal (27.4 kW)


class ConstraintBuilder:
    """Builder for MILP constraints."""

    def __init__(
        self,
        hours: int = 8760,
        bounds: Optional[CapacityBounds] = None,
    ):
        """Initialize constraint builder.

        Args:
            hours: Number of hours in simulation
            bounds: Capacity bounds for components
        """
        self.hours = hours
        self.bounds = bounds or CapacityBounds()

    def add_power_balance_constraint(
        self,
        model: pulp.LpProblem,
        p_pv: List[pulp.LpVariable],
        p_wind: List[pulp.LpVariable],
        p_fc: List[pulp.LpVariable],
        p_bm: List[pulp.LpVariable],
        p_elz: List[pulp.LpVariable],
        p_ume: List[pulp.LpVariable],
        demand: np.ndarray,
    ) -> None:
        """Add power balance constraint (Equation 8).

        P^HRES_h = P^D_h + P^ELZ_h
        P^PV + P^WT + P^FC + P^BM = P^D + P^ELZ + UME

        Args:
            model: PuLP model
            p_pv: PV power variables
            p_wind: Wind power variables
            p_fc: Fuel cell power variables
            p_bm: Biomass power variables
            p_elz: Electrolyzer consumption variables
            p_ume: Unmet energy variables
            demand: Hourly demand array
        """
        for h in range(self.hours):
            model += (
                p_pv[h] + p_wind[h] + p_fc[h] + p_bm[h]
                == demand[h] + p_elz[h] - p_ume[h]
            ), f"power_balance_{h}"

    def add_dispatch_mode_constraints(
        self,
        model: pulp.LpProblem,
        y_fc: List[pulp.LpVariable],
        y_bm: List[pulp.LpVariable],
        mode_0: List[pulp.LpVariable],
        mode_a: List[pulp.LpVariable],
        mode_b: List[pulp.LpVariable],
        mode_c: List[pulp.LpVariable],
    ) -> None:
        """Add dispatch mode selection constraints (Equations 9-10).

        Exactly one mode must be active at each hour:
        ℶ_0 + ℶ_a + ℶ_b + ℶ_c = 1

        Mode definitions:
        - ℶ_0: Renewables only (excess to electrolyzer)
        - ℶ_a: Renewables + Fuel Cell
        - ℶ_b: Renewables + Biomass
        - ℶ_c: Renewables + Fuel Cell + Biomass

        Args:
            model: PuLP model
            y_fc: Fuel cell on/off variables
            y_bm: Biomass on/off variables
            mode_0, mode_a, mode_b, mode_c: Mode selection variables
        """
        for h in range(self.hours):
            # Exactly one mode active
            model += (
                mode_0[h] + mode_a[h] + mode_b[h] + mode_c[h] == 1
            ), f"mode_selection_{h}"

            # Link modes to component status
            # Mode 0: FC=0, BM=0
            model += y_fc[h] >= mode_a[h], f"fc_mode_a_{h}"
            model += y_fc[h] >= mode_c[h], f"fc_mode_c_{h}"
            model += y_fc[h] <= mode_a[h] + mode_c[h], f"fc_mode_limit_{h}"

            model += y_bm[h] >= mode_b[h], f"bm_mode_b_{h}"
            model += y_bm[h] >= mode_c[h], f"bm_mode_c_{h}"
            model += y_bm[h] <= mode_b[h] + mode_c[h], f"bm_mode_limit_{h}"

    def add_capacity_constraints(
        self,
        model: pulp.LpProblem,
        cap_pv: pulp.LpVariable,
        cap_wind: pulp.LpVariable,
        cap_elz: pulp.LpVariable,
        cap_fc: pulp.LpVariable,
        cap_h2: pulp.LpVariable,
        cap_bm: pulp.LpVariable,
    ) -> None:
        """Add capacity bound constraints.

        Args:
            model: PuLP model
            cap_*: Capacity decision variables
        """
        model += cap_pv >= self.bounds.pv_min, "pv_min"
        model += cap_pv <= self.bounds.pv_max, "pv_max"

        model += cap_wind >= self.bounds.wind_min, "wind_min"
        model += cap_wind <= self.bounds.wind_max, "wind_max"

        model += cap_elz >= self.bounds.electrolyzer_min, "elz_min"
        model += cap_elz <= self.bounds.electrolyzer_max, "elz_max"

        model += cap_fc >= self.bounds.fuel_cell_min, "fc_min"
        model += cap_fc <= self.bounds.fuel_cell_max, "fc_max"

        model += cap_h2 >= self.bounds.h2_storage_min, "h2_min"
        model += cap_h2 <= self.bounds.h2_storage_max, "h2_max"

        model += cap_bm >= self.bounds.biomass_min, "bm_min"
        model += cap_bm <= self.bounds.biomass_max, "bm_max"

    def add_pv_output_constraints(
        self,
        model: pulp.LpProblem,
        p_pv: List[pulp.LpVariable],
        cap_pv: pulp.LpVariable,
        irradiance_factor: np.ndarray,
    ) -> None:
        """Add PV output constraints.

        P^PV_h <= Cap_PV * irradiance_factor_h

        Args:
            model: PuLP model
            p_pv: PV power variables
            cap_pv: PV capacity variable
            irradiance_factor: Normalized irradiance (0-1)
        """
        for h in range(self.hours):
            model += p_pv[h] <= cap_pv * irradiance_factor[h], f"pv_output_{h}"
            model += p_pv[h] >= 0, f"pv_nonneg_{h}"

    def add_wind_output_constraints(
        self,
        model: pulp.LpProblem,
        p_wind: List[pulp.LpVariable],
        cap_wind: pulp.LpVariable,
        wind_factor: np.ndarray,
    ) -> None:
        """Add wind output constraints.

        P^WT_h <= Cap_WT * wind_factor_h

        Args:
            model: PuLP model
            p_wind: Wind power variables
            cap_wind: Wind capacity variable
            wind_factor: Capacity factor from wind curve (0-1)
        """
        for h in range(self.hours):
            model += p_wind[h] <= cap_wind * wind_factor[h], f"wind_output_{h}"
            model += p_wind[h] >= 0, f"wind_nonneg_{h}"

    def add_electrolyzer_constraints(
        self,
        model: pulp.LpProblem,
        p_elz: List[pulp.LpVariable],
        y_elz: List[pulp.LpVariable],
        cap_elz: pulp.LpVariable,
        min_load_fraction: float = 0.1,
    ) -> None:
        """Add electrolyzer operating constraints.

        Minimum load: P^ELZ >= min_load * Cap * y_elz
        Maximum load: P^ELZ <= Cap * y_elz

        Args:
            model: PuLP model
            p_elz: Electrolyzer power variables
            y_elz: Electrolyzer on/off variables
            cap_elz: Electrolyzer capacity variable
            min_load_fraction: Minimum operating load fraction
        """
        M = self.bounds.electrolyzer_max  # Big-M value

        for h in range(self.hours):
            # Upper bound when on
            model += p_elz[h] <= cap_elz, f"elz_max_{h}"

            # Link to on/off status
            model += p_elz[h] <= M * y_elz[h], f"elz_on_{h}"

            # Minimum load when on
            model += (
                p_elz[h] >= min_load_fraction * cap_elz - M * (1 - y_elz[h])
            ), f"elz_min_load_{h}"

    def add_fuel_cell_constraints(
        self,
        model: pulp.LpProblem,
        p_fc: List[pulp.LpVariable],
        y_fc: List[pulp.LpVariable],
        cap_fc: pulp.LpVariable,
    ) -> None:
        """Add fuel cell operating constraints.

        Args:
            model: PuLP model
            p_fc: Fuel cell power variables
            y_fc: Fuel cell on/off variables
            cap_fc: Fuel cell capacity variable
        """
        M = self.bounds.fuel_cell_max

        for h in range(self.hours):
            model += p_fc[h] <= cap_fc, f"fc_max_{h}"
            model += p_fc[h] <= M * y_fc[h], f"fc_on_{h}"
            model += p_fc[h] >= 0, f"fc_nonneg_{h}"

    def add_biomass_constraints(
        self,
        model: pulp.LpProblem,
        p_bm: List[pulp.LpVariable],
        y_bm: List[pulp.LpVariable],
        cap_bm: pulp.LpVariable,
        min_load_fraction: float = 0.3,
    ) -> None:
        """Add biomass generator operating constraints.

        Args:
            model: PuLP model
            p_bm: Biomass power variables
            y_bm: Biomass on/off variables
            cap_bm: Biomass capacity variable
            min_load_fraction: Minimum operating load fraction
        """
        M = self.bounds.biomass_max

        for h in range(self.hours):
            model += p_bm[h] <= cap_bm, f"bm_max_{h}"
            model += p_bm[h] <= M * y_bm[h], f"bm_on_{h}"
            model += (
                p_bm[h] >= min_load_fraction * cap_bm - M * (1 - y_bm[h])
            ), f"bm_min_load_{h}"

    def add_hydrogen_storage_constraints(
        self,
        model: pulp.LpProblem,
        h2_level: List[pulp.LpVariable],
        q_elz: List[pulp.LpVariable],
        q_fc: List[pulp.LpVariable],
        g_h: List[pulp.LpVariable],
        cap_h2: pulp.LpVariable,
        soc_min: float = 0.1,
        soc_max: float = 0.95,
        initial_soc: float = 0.5,
    ) -> None:
        """Add hydrogen storage balance constraints (Equation 16).

        H_h = H_{h-1} + Q^ELZ - Q^FC - G^H

        Args:
            model: PuLP model
            h2_level: H2 storage level variables
            q_elz: H2 from electrolyzer variables
            q_fc: H2 to fuel cell variables
            g_h: H2 sold to market variables
            cap_h2: H2 storage capacity variable
            soc_min, soc_max: SOC limits
            initial_soc: Initial SOC
        """
        for h in range(self.hours):
            # Storage dynamics
            if h == 0:
                model += (
                    h2_level[h] == initial_soc * cap_h2 + q_elz[h] - q_fc[h] - g_h[h]
                ), f"h2_balance_{h}"
            else:
                model += (
                    h2_level[h] == h2_level[h - 1] + q_elz[h] - q_fc[h] - g_h[h]
                ), f"h2_balance_{h}"

            # SOC limits
            model += h2_level[h] >= soc_min * cap_h2, f"h2_soc_min_{h}"
            model += h2_level[h] <= soc_max * cap_h2, f"h2_soc_max_{h}"

    def add_feedstock_constraint(
        self,
        model: pulp.LpProblem,
        feedstock_consumption: List[pulp.LpVariable],
        annual_availability_ton: float,
    ) -> None:
        """Add feedstock availability constraint (Equation 32).

        Σ B_h ≤ B_fdT

        Args:
            model: PuLP model
            feedstock_consumption: Hourly feedstock consumption
            annual_availability_ton: Annual feedstock availability in tons
        """
        # Convert to kg/h limit
        total_consumption = pulp.lpSum(feedstock_consumption)
        model += total_consumption <= annual_availability_ton * 1000, "feedstock_limit"

    def add_epsilon_constraint(
        self,
        model: pulp.LpProblem,
        ume_expression: pulp.LpAffineExpression,
        epsilon: float,
    ) -> None:
        """Add ε-constraint for multi-objective optimization.

        f2(UME) ≤ ε

        Args:
            model: PuLP model
            ume_expression: UME objective expression
            epsilon: Upper bound on UME
        """
        model += ume_expression <= epsilon, "epsilon_constraint"

    def validate_solution(
        self,
        p_pv: np.ndarray,
        p_wind: np.ndarray,
        p_fc: np.ndarray,
        p_bm: np.ndarray,
        p_elz: np.ndarray,
        demand: np.ndarray,
        h2_level: np.ndarray,
        cap_h2: float,
    ) -> Dict[str, bool]:
        """Validate solution against all constraints.

        Args:
            Power profiles and demand array
            h2_level: H2 storage level profile
            cap_h2: H2 storage capacity

        Returns:
            Dict with validation results for each constraint
        """
        supply = p_pv + p_wind + p_fc + p_bm
        total_demand = demand + p_elz

        return {
            "power_balance": np.allclose(supply, total_demand, atol=0.1),
            "pv_nonneg": np.all(p_pv >= -0.001),
            "wind_nonneg": np.all(p_wind >= -0.001),
            "fc_nonneg": np.all(p_fc >= -0.001),
            "bm_nonneg": np.all(p_bm >= -0.001),
            "elz_nonneg": np.all(p_elz >= -0.001),
            "h2_soc_min": np.all(h2_level >= 0.1 * cap_h2 - 0.1),
            "h2_soc_max": np.all(h2_level <= 0.95 * cap_h2 + 0.1),
        }
