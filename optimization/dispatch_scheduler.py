"""Dispatch scheduler implementing the operational logic.

Reference: Section 2.5.2.1, Equations 8-10, Figure 4

Corrected two-branch flowchart logic (h2_hres_corrected_flowchart.html):

Decision 1: P^PV + P^WT >= P^D ?

LEFT BRANCH (YES — Surplus S₁, ℶ₀=1):
    Excess → Electrolyzer → H₂ production
    Decision 2: H_h > H_min ?
      YES → β Decision (optimizer chooses):
        β=1: Sell surplus H₂, Biomass backup, FC OFF  → Mode B (ℶ_b=1)
        β=0: FC from stored H₂, no H₂ sold            → Mode A (ℶₐ=1)
             if still deficit → Biomass too            → Mode C (ℶ_c=1)
      NO  → Biomass fallback only (no FC, no H₂ sold) → Mode B (ℶ_b=1)

RIGHT BRANCH (NO — Deficit S₂, β is not evaluated):
    No H₂ sales (all H₂ reserved for FC)
    Activate FC first
    Decision 3b: P^PV + P^WT + P^FC >= P^D ?
      YES → Mode A (ℶₐ=1)
      NO  → Also activate Biomass
            Decision 3c: P^PV + P^WT + P^FC + P^BM >= P^D ?
              YES → Mode C (ℶ_c=1)
              NO  → Record UME

Seasonal β heuristic (used when beta_profile not supplied):
    Dry months (Jan, Feb, Jun, Jul, Aug, Sep): β=1 (sell H₂, biomass backup)
    Wet months (Mar, Apr, May, Oct, Nov, Dec): β=0 (FC backup, no H₂ sale)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from components import (
    SolarPV,
    WindTurbine,
    Electrolyzer,
    FuelCell,
    HydrogenStorage,
    BiomassGenerator,
)


class DispatchMode(Enum):
    """Dispatch modes from Equation 9."""

    MODE_0 = auto()  # Renewables only (surplus, electrolyzer active)
    MODE_A = auto()  # Renewables + Fuel Cell
    MODE_B = auto()  # Renewables + Biomass
    MODE_C = auto()  # Renewables + FC + Biomass


@dataclass
class HourlyDispatch:
    """Dispatch result for a single hour."""

    mode: DispatchMode
    p_pv: float     # kW
    p_wind: float   # kW
    p_fc: float     # kW
    p_bm: float     # kW
    p_elz: float    # kW (consumption)
    p_ume: float    # kW (unmet)
    h2_produced: float  # kg
    h2_consumed: float  # kg
    h2_sold: float      # kg
    h2_level: float     # kg
    feedstock_used: float  # kg
    beta: int           # 0 or 1 — the β decision for this hour


@dataclass
class DispatchResult:
    """Complete dispatch result over simulation period."""

    hourly: List[HourlyDispatch]
    mode_counts: Dict[DispatchMode, int]
    total_energy_served: float
    total_unmet_energy: float
    total_h2_produced: float
    total_h2_consumed: float
    total_h2_sold: float
    total_feedstock_used: float


# Nairobi seasonal β heuristic (paper Table 3-4 / seasonal analysis)
# Months where solar+wind surplus is high → sell H₂ (β=1)
_DRY_MONTHS = {1, 2, 6, 7, 8, 9}   # Jan, Feb, Jun-Sep


def _seasonal_beta(hour_of_year: int) -> int:
    """Return β=1 for dry season months, β=0 for wet season months.

    Args:
        hour_of_year: Hour index (0-8759)

    Returns:
        1 for dry season (sell H₂), 0 for wet season (FC backup)
    """
    import pandas as pd
    dt = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=hour_of_year)
    return 1 if dt.month in _DRY_MONTHS else 0


class DispatchScheduler:
    """Scheduler implementing corrected dispatch logic from Figure 4.

    The two-branch flowchart is implemented in dispatch_hour().
    β (beta) controls whether surplus H₂ is sold (β=1) or reserved
    for the Fuel Cell (β=0) on the LEFT (surplus) branch.
    On the RIGHT (deficit) branch, β is irrelevant — H₂ is always
    reserved for power generation.
    """

    def __init__(
        self,
        pv: SolarPV,
        wind: WindTurbine,
        electrolyzer: Electrolyzer,
        fuel_cell: FuelCell,
        h2_storage: HydrogenStorage,
        biomass: BiomassGenerator,
    ):
        """Initialize dispatch scheduler.

        Args:
            All component instances
        """
        self.pv = pv
        self.wind = wind
        self.electrolyzer = electrolyzer
        self.fuel_cell = fuel_cell
        self.h2_storage = h2_storage
        self.biomass = biomass

    def dispatch_hour(
        self,
        demand: float,
        irradiance: float,
        temperature: float,
        wind_speed: float,
        lhv_mj_kg: float,
        feedstock_available: float = float("inf"),
        beta: int = 1,
    ) -> HourlyDispatch:
        """Dispatch power for a single hour following the corrected flowchart.

        Args:
            demand: Load demand this hour (kW)
            irradiance: Solar irradiance (W/m²)
            temperature: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            lhv_mj_kg: Biomass lower heating value for this hour (MJ/kg)
            feedstock_available: Remaining biomass feedstock (kg)
            beta: β decision for H₂ market on surplus branch.
                  1 = sell surplus H₂ + biomass backup (dry season),
                  0 = FC backup from stored H₂, no H₂ sold (wet season).

        Returns:
            HourlyDispatch with all power flows and H₂ levels.
        """
        # ── Step 1: Calculate renewable generation ──────────────────────────
        pv_output = self.pv.calculate_output(irradiance, temperature)
        p_pv = float(np.atleast_1d(pv_output.power_kw)[0])

        wind_output = self.wind.calculate_output(wind_speed)
        p_wind = float(np.atleast_1d(wind_output.power_kw)[0])

        renewable_power = p_pv + p_wind   # P^PV + P^WT

        # ── Initialise all outputs to zero ──────────────────────────────────
        p_bm = 0.0
        p_fc = 0.0
        p_elz = 0.0
        p_ume = 0.0
        h2_produced = 0.0
        h2_consumed = 0.0
        h2_sold = 0.0
        feedstock_used = 0.0
        mode = DispatchMode.MODE_0

        # H₂ storage state
        h2_available = self.h2_storage.available_to_discharge  # above SOC_min

        # ════════════════════════════════════════════════════════════════════
        # DECISION 1: P^PV + P^WT >= P^D  ?
        # ════════════════════════════════════════════════════════════════════
        if renewable_power >= demand:
            # ────────────────────────────────────────────────────────────────
            # LEFT BRANCH — Surplus (S₁)
            # Demand is met by renewables; excess feeds the electrolyzer.
            # ────────────────────────────────────────────────────────────────
            mode = DispatchMode.MODE_0   # ℶ₀ = 1

            excess = renewable_power - demand

            # Run electrolyzer on excess renewable power
            if excess > 0:
                elz_power = min(excess, self.electrolyzer.capacity)
                min_load = (
                    self.electrolyzer.capacity
                    * self.electrolyzer.params.min_load_fraction
                )
                if elz_power >= min_load:
                    p_elz = elz_power
                    elz_output = self.electrolyzer.calculate_output(p_elz)
                    h2_produced = float(
                        np.atleast_1d(elz_output.details["h2_production_kg_h"])[0]
                    )

            # DECISION 2: H_h > H_min?  (precondition for β decision)
            h2_above_min = h2_available > 0  # available_to_discharge already above min

            if h2_above_min:
                # ── β DECISION ──────────────────────────────────────────────
                if beta == 1:
                    # β = 1 path: Sell H₂ from storage up to market rate, FC OFF
                    # (Dry season — abundant H₂ and renewables)
                    # Sell from available stored H₂ up to the hourly market rate limit.
                    # This matches the MILP constraint h2_max_sales_rate = 0.5 kg/h.
                    # New production (h2_produced) goes into storage FIRST via simulate_hour;
                    # we sell from the current available balance — NOT just overflow.
                    H2_MARKET_RATE = 0.5  # kg/hour (mirrors CapacityBounds.h2_max_sales_rate)
                    h2_sold = min(h2_available, H2_MARKET_RATE)

                    # Biomass as backup for reliability
                    # On the surplus side demand is already covered by renewables,
                    # so biomass is called only if there is any remaining gap
                    # (e.g. electrolyzer draws some extra load).
                    gap = demand - renewable_power   # ≤ 0 on surplus side
                    if gap > 0 and self.biomass.capacity > 0:
                        bm_target = min(self.biomass.capacity, gap)
                        bm_output = self.biomass.calculate_output(
                            bm_target, lhv_mj_kg, feedstock_available
                        )
                        p_bm = float(np.atleast_1d(bm_output.power_kw)[0])
                        feedstock_used = float(
                            np.atleast_1d(bm_output.details["feed_rate_kg_h"])[0]
                        )
                        mode = DispatchMode.MODE_B  # ℶ_b = 1

                        total_supply = renewable_power + p_bm
                        if total_supply < demand:
                            p_ume = demand - total_supply
                    # else: no gap, demand fully met by renewables → MODE_0

                else:
                    # β = 0 path: FC backup from stored H₂, no H₂ sold
                    # (Wet season — low renewables, H₂ needed for power)
                    h2_sold = 0.0

                    # FC provides supplemental power
                    gap = demand - renewable_power   # ≤ 0 on surplus side
                    if gap > 0 and h2_available > 0:
                        target_fc = min(self.fuel_cell.capacity, gap)
                        fc_output = self.fuel_cell.calculate_output(
                            target_fc, h2_available
                        )
                        p_fc = float(np.atleast_1d(fc_output.power_kw)[0])
                        h2_consumed = float(
                            np.atleast_1d(fc_output.details["h2_consumption_kg_h"])[0]
                        )

                        if renewable_power + p_fc >= demand:
                            mode = DispatchMode.MODE_A  # ℶₐ = 1
                        else:
                            # Add biomass if FC alone not enough
                            remaining = demand - renewable_power - p_fc
                            if self.biomass.capacity > 0:
                                bm_target = min(self.biomass.capacity, remaining)
                                bm_output = self.biomass.calculate_output(
                                    bm_target, lhv_mj_kg, feedstock_available
                                )
                                p_bm = float(np.atleast_1d(bm_output.power_kw)[0])
                                feedstock_used = float(
                                    np.atleast_1d(bm_output.details["feed_rate_kg_h"])[0]
                                )
                            mode = DispatchMode.MODE_C  # ℶ_c = 1
                            total = renewable_power + p_fc + p_bm
                            if total < demand:
                                p_ume = demand - total
                    # else: no gap, demand met by renewables → MODE_0

            else:
                # H₂ ≤ H_min: Biomass fallback only (no FC, no H₂ sold)
                h2_sold = 0.0
                gap = demand - renewable_power
                if gap > 0 and self.biomass.capacity > 0:
                    bm_target = min(self.biomass.capacity, gap)
                    bm_output = self.biomass.calculate_output(
                        bm_target, lhv_mj_kg, feedstock_available
                    )
                    p_bm = float(np.atleast_1d(bm_output.power_kw)[0])
                    feedstock_used = float(
                        np.atleast_1d(bm_output.details["feed_rate_kg_h"])[0]
                    )
                    mode = DispatchMode.MODE_B
                    if renewable_power + p_bm < demand:
                        p_ume = demand - renewable_power - p_bm

        else:
            # ────────────────────────────────────────────────────────────────
            # RIGHT BRANCH — Deficit (S₂)
            # PV + WT cannot meet demand. β is NOT evaluated.
            # All H₂ is reserved for FC power — no market sales.
            # ────────────────────────────────────────────────────────────────
            h2_sold = 0.0   # No H₂ sales on the deficit side
            deficit = demand - renewable_power

            # ── FC first (primary backup) ──────────────────────────────────
            if h2_available > 0:
                target_fc = min(deficit, self.fuel_cell.capacity)
                fc_output = self.fuel_cell.calculate_output(target_fc, h2_available)
                p_fc = float(np.atleast_1d(fc_output.power_kw)[0])
                h2_consumed = float(
                    np.atleast_1d(fc_output.details["h2_consumption_kg_h"])[0]
                )

            # Decision 3b: PV + WT + FC >= demand?
            if renewable_power + p_fc >= demand:
                mode = DispatchMode.MODE_A   # ℶₐ = 1
                p_ume = max(demand - renewable_power - p_fc, 0.0)

            else:
                # FC not sufficient → Activate Biomass too
                remaining = demand - renewable_power - p_fc
                if self.biomass.capacity > 0:
                    bm_target = min(self.biomass.capacity, remaining)
                    bm_output = self.biomass.calculate_output(
                        bm_target, lhv_mj_kg, feedstock_available
                    )
                    p_bm = float(np.atleast_1d(bm_output.power_kw)[0])
                    feedstock_used = float(
                        np.atleast_1d(bm_output.details["feed_rate_kg_h"])[0]
                    )

                # Decision 3c: PV + WT + FC + BM >= demand?
                mode = DispatchMode.MODE_C   # ℶ_c = 1
                total_supply = renewable_power + p_fc + p_bm
                if total_supply < demand:
                    p_ume = demand - total_supply

        # ── Update H₂ storage (Equation 16) ────────────────────────────────
        storage_result = self.h2_storage.simulate_hour(
            h2_from_elz=h2_produced,
            h2_to_fc=h2_consumed,
            h2_to_market=h2_sold,
        )
        h2_level = self.h2_storage.h2_stored_kg

        return HourlyDispatch(
            mode=mode,
            p_pv=p_pv,
            p_wind=p_wind,
            p_fc=p_fc,
            p_bm=p_bm,
            p_elz=p_elz,
            p_ume=p_ume,
            h2_produced=h2_produced,
            h2_consumed=h2_consumed,
            h2_sold=storage_result["h2_to_market_kg"],
            h2_level=h2_level,
            feedstock_used=feedstock_used,
            beta=beta,
        )

    def simulate_year(
        self,
        demand_profile: np.ndarray,
        irradiance_profile: np.ndarray,
        temperature_profile: np.ndarray,
        wind_speed_profile: np.ndarray,
        lhv_profile: np.ndarray,
        annual_feedstock_kg: float = float("inf"),
        beta_profile: Optional[np.ndarray] = None,
    ) -> DispatchResult:
        """Simulate dispatch for a full year.

        Args:
            demand_profile: Hourly demand (8760 values, kW)
            irradiance_profile: Hourly irradiance (W/m²)
            temperature_profile: Hourly temperature (°C)
            wind_speed_profile: Hourly wind speed (m/s)
            lhv_profile: Hourly biomass LHV (MJ/kg)
            annual_feedstock_kg: Total feedstock available for year
            beta_profile: Optional array of β values (0 or 1) for each hour.
                          If None, the seasonal heuristic is applied
                          (β=1 in dry months, β=0 in wet months).

        Returns:
            DispatchResult with all hourly results and totals
        """
        # Reset storage to initial state
        self.h2_storage.reset()

        hours = len(demand_profile)

        # Build β profile if not supplied
        if beta_profile is None:
            beta_profile = np.array([_seasonal_beta(h) for h in range(hours)])

        # Track feedstock consumption
        feedstock_remaining = annual_feedstock_kg

        hourly_results = []
        mode_counts = {mode: 0 for mode in DispatchMode}

        for h in range(hours):
            result = self.dispatch_hour(
                demand=demand_profile[h],
                irradiance=irradiance_profile[h],
                temperature=temperature_profile[h],
                wind_speed=wind_speed_profile[h],
                lhv_mj_kg=lhv_profile[h],
                feedstock_available=feedstock_remaining,
                beta=int(beta_profile[h]),
            )

            hourly_results.append(result)
            mode_counts[result.mode] += 1
            feedstock_remaining -= result.feedstock_used

        # Calculate totals
        total_energy_served = sum(
            r.p_pv + r.p_wind + r.p_fc + r.p_bm - r.p_elz for r in hourly_results
        )
        total_unmet = sum(r.p_ume for r in hourly_results)
        total_h2_produced = sum(r.h2_produced for r in hourly_results)
        total_h2_consumed = sum(r.h2_consumed for r in hourly_results)
        total_h2_sold = sum(r.h2_sold for r in hourly_results)
        total_feedstock = sum(r.feedstock_used for r in hourly_results)

        return DispatchResult(
            hourly=hourly_results,
            mode_counts=mode_counts,
            total_energy_served=total_energy_served,
            total_unmet_energy=total_unmet,
            total_h2_produced=total_h2_produced,
            total_h2_consumed=total_h2_consumed,
            total_h2_sold=total_h2_sold,
            total_feedstock_used=total_feedstock,
        )

    def get_mode_statistics(
        self,
        result: DispatchResult,
    ) -> Dict[str, float]:
        """Get statistics about dispatch mode usage.

        Args:
            result: DispatchResult from simulation

        Returns:
            Dict with mode usage statistics
        """
        total_hours = len(result.hourly)

        beta_1_hours = sum(1 for r in result.hourly if r.beta == 1)
        beta_0_hours = total_hours - beta_1_hours

        return {
            "mode_0_hours": result.mode_counts[DispatchMode.MODE_0],
            "mode_0_pct": result.mode_counts[DispatchMode.MODE_0] / total_hours * 100,
            "mode_a_hours": result.mode_counts[DispatchMode.MODE_A],
            "mode_a_pct": result.mode_counts[DispatchMode.MODE_A] / total_hours * 100,
            "mode_b_hours": result.mode_counts[DispatchMode.MODE_B],
            "mode_b_pct": result.mode_counts[DispatchMode.MODE_B] / total_hours * 100,
            "mode_c_hours": result.mode_counts[DispatchMode.MODE_C],
            "mode_c_pct": result.mode_counts[DispatchMode.MODE_C] / total_hours * 100,
            "beta_1_hours": beta_1_hours,
            "beta_0_hours": beta_0_hours,
        }
