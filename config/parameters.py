"""System parameters and component costs from the paper.

Reference: Mulumba & Farzaneh (2025) - Table 2, Section 3.1
International Journal of Hydrogen Energy 178 (2025) 151474
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComponentLifetimes:
    """Component lifetimes in years for replacement cost calculation."""
    
    pv: int = 25
    wind: int = 20
    electrolyzer: int = 15
    fuel_cell: int = 5
    h2_tank: int = 25
    biomass: int = 20

@dataclass
class ComponentCosts:
    """Component capital and O&M costs from Table 2.

    All capital costs in $/kW (or $/kg for H2 tank).
    O&M costs vary by component type.
    """

    # Capital costs ($/kW)
    pv_capital: float = 900.0  # $/kW
    wind_capital: float = 1200.0  # $/kW
    electrolyzer_capital: float = 890.0  # $/kW
    fuel_cell_capital: float = 1000.0  # $/kW
    h2_tank_capital: float = 1100.0  # $/kg capacity
    biomass_capital: float = 600.0  # $/kW

    # O&M costs
    pv_om_annual: float = 55.0  # $/year per kW
    wind_om_annual: float = 41.78  # $/year per kW
    electrolyzer_om_annual: float = 20.0  # $/year per kW
    fuel_cell_om_hourly: float = 0.01  # $/hour of operation
    h2_tank_om_annual: float = 0.0  # $/year
    biomass_om_annual: float = 15.0  # $/year per kW
    
    # Fuel costs
    biomass_fuel_cost: float = 40.0  # $/ton


@dataclass
class EconomicParameters:
    """Economic parameters for COE calculation.

    Reference: Section 2.5.1, Equation 2
    """

    # Discount rate (12% as stated in Section 3.1)
    discount_rate: float = 0.12

    # Project lifetime (years)
    project_lifetime: int = 25

    # Hydrogen market prices ($/kg)
    h2_price_base: float = 6.6  # Base case
    h2_price_high: float = 9.9  # Sensitivity analysis

    # Inflation rate (assumed)
    inflation_rate: float = 0.03

    # Installation cost factor — multiplies all capital costs.
    # Table 2 lists equipment costs; total installed cost for off-grid Kenya
    # includes BOS, inverters, wiring, civil works, transport, labor (~2x).
    # Verified: inst=2.0 reproduces paper's COE ($0.494 with H2, $0.668 without).
    installation_factor: float = 2.0

    def capital_recovery_factor(self) -> float:
        """Calculate the Capital Recovery Factor (CRF).

        CRF = dr * (1 + dr)^n / ((1 + dr)^n - 1)
        """
        dr = self.discount_rate
        n = self.project_lifetime
        return (dr * (1 + dr) ** n) / ((1 + dr) ** n - 1)


@dataclass
class PVParameters:
    """Solar PV system parameters for Equation 11."""

    # PV derating factor (accounts for losses)
    derating_factor: float = 0.80

    # Temperature coefficient of power (%/°C)
    temp_coefficient: float = -0.0045

    # Panel efficiency at STC
    efficiency_stc: float = 0.20

    # Maximum capacity bounds for optimization (kW)
    capacity_min: float = 0.0
    capacity_max: float = 100.0


@dataclass
class WindParameters:
    """Wind turbine parameters for Equation 12."""

    # Cut-in wind speed (m/s)
    v_cut_in: float = 3.0

    # Rated wind speed (m/s)
    v_rated: float = 12.0

    # Cut-out wind speed (m/s)
    v_cut_out: float = 25.0

    # Hub height (m)
    hub_height: float = 50.0

    # Reference height for wind data (m)
    reference_height: float = 10.0

    # Wind shear exponent (typical for open terrain)
    wind_shear_exponent: float = 0.14

    # Maximum capacity bounds for optimization (kW)
    capacity_min: float = 0.0
    capacity_max: float = 100.0


@dataclass
class ElectrolyzerParameters:
    """PEM Electrolyzer parameters for Equation 13.

    Reference: Section 2.5.2.3, Supplementary S2
    """

    # Overall efficiency (voltage × Faradaic × auxiliary)
    efficiency: float = 0.70

    # Minimum operating load (% of rated)
    min_load_fraction: float = 0.10

    # Maximum capacity bounds (kW)
    capacity_min: float = 0.0
    capacity_max: float = 100.0

    # Stack temperature (°C)
    operating_temp: float = 80.0


@dataclass
class FuelCellParameters:
    """PEM Fuel Cell parameters for Equations 14-15.

    Reference: Section 2.5.2.3, Supplementary S1
    """

    # Number of cells in stack
    n_cells: int = 100

    # Active cell area (cm²)
    cell_area: float = 200.0

    # Operating pressure (bar)
    operating_pressure: float = 2.0

    # Operating temperature (°C)
    operating_temp: float = 80.0

    # Maximum capacity bounds (kW)
    capacity_min: float = 0.0
    capacity_max: float = 50.0


@dataclass
class HydrogenStorageParameters:
    """Hydrogen storage tank parameters for Equation 16."""

    # Minimum state of charge (fraction)
    soc_min: float = 0.10

    # Maximum state of charge (fraction)
    soc_max: float = 0.95

    # Storage efficiency (compression losses)
    storage_efficiency: float = 0.98

    # Maximum capacity bounds (kg)
    capacity_min: float = 0.0
    capacity_max: float = 500.0

    # Initial storage level (fraction of capacity)
    initial_soc: float = 0.50


@dataclass
class BiomassParameters:
    """Biomass generator parameters for Equations 17-21, 28.

    Reference: Section 2.5.2.4, Tables 3-4
    """

    # Steam Rankine cycle efficiency
    thermal_efficiency: float = 0.35

    # Minimum operating load (fraction)
    min_load_fraction: float = 0.30

    # Heat losses (Equation 21 components)
    loss_fuel_moisture: float = 0.05  # e_fm
    loss_unburned: float = 0.02  # e_uc
    loss_dry_gas: float = 0.08  # e_dg
    loss_latent_heat: float = 0.03  # e_lh
    loss_moisture_air: float = 0.01  # e_ma
    loss_manufacturing: float = 0.02  # e_mfc

    # LHV range (MJ/kg) based on moisture content
    lhv_dry: float = 17.5  # Dry season (0-12 mm precipitation)
    lhv_wet: float = 14.3  # Wet season (~120 mm precipitation)

    # Feedstock mix ratio (wood:grass = 1:10)
    wood_fraction: float = 0.091  # 1/11
    grass_fraction: float = 0.909  # 10/11

    # Annual feedstock availability (ton/year) from Table 3
    grass_available: float = 19_315_584.0
    wood_available: float = 1_622_350.0

    # Maximum capacity bounds (kW)
    capacity_min: float = 0.0
    capacity_max: float = 100.0

    @property
    def total_heat_loss(self) -> float:
        """Calculate total heat loss factor (e_Ls) per Equation 21."""
        return (
            self.loss_fuel_moisture
            + self.loss_unburned
            + self.loss_dry_gas
            + self.loss_latent_heat
            + self.loss_moisture_air
            + self.loss_manufacturing
        )


@dataclass
class LoadParameters:
    """Load profile parameters.

    Reference: Section 3.1
    """

    # Total annual demand (kWh)
    annual_demand: float = 127_800.0  # 127.8 MWh

    # Number of households
    n_households: int = 70

    # Number of buildings
    n_buildings: int = 3

    # Peak hours (0-23)
    morning_peak_start: int = 8
    morning_peak_end: int = 11
    evening_peak_start: int = 20
    evening_peak_end: int = 22

    # Peak to base load ratio
    peak_to_base_ratio: float = 2.5


@dataclass
class SystemParameters:
    """Aggregated system parameters."""

    costs: ComponentCosts = field(default_factory=ComponentCosts)
    lifetimes: ComponentLifetimes = field(default_factory=ComponentLifetimes)
    economic: EconomicParameters = field(default_factory=EconomicParameters)
    pv: PVParameters = field(default_factory=PVParameters)
    wind: WindParameters = field(default_factory=WindParameters)
    electrolyzer: ElectrolyzerParameters = field(default_factory=ElectrolyzerParameters)
    fuel_cell: FuelCellParameters = field(default_factory=FuelCellParameters)
    h2_storage: HydrogenStorageParameters = field(
        default_factory=HydrogenStorageParameters
    )
    biomass: BiomassParameters = field(default_factory=BiomassParameters)
    load: LoadParameters = field(default_factory=LoadParameters)


# Default instance for easy import
DEFAULT_PARAMS = SystemParameters()
