"""Parameters for the NSGA-II paper replication.

Reference: Ahmed et al. (2024)
"Energy management and sizing of a stand-alone hybrid renewable energy
system for community electricity, fresh water, and cooking gas demands
of a remote island"
Energy Conversion and Management 299 (2024) 117865

All values from Table 3 and paper equations.
"""

from dataclasses import dataclass, field


@dataclass
class NSGAPVParams:
    """PV module parameters (Table 3, Eq 13-14)."""
    derating_factor: float = 0.90
    temp_coefficient: float = -0.0043  # /°C
    efficiency: float = 0.204  # 20.4%
    noct: float = 45.0  # °C
    transmittance_absorptance: float = 0.90  # τα
    capital_cost: float = 1300.0  # $/kW
    om_cost: float = 20.0  # $/kW-year
    lifetime: int = 25  # years


@dataclass
class NSGAWindParams:
    """Wind turbine parameters (Table 3, Eq 15-17)."""
    v_cut_in: float = 2.75  # m/s
    v_rated: float = 6.5  # m/s
    v_cut_off: float = 20.0  # m/s (furling)
    hub_height: float = 20.0  # m
    reference_height: float = 10.0  # m
    wind_shear_exponent: float = 0.14
    weibull_shape: float = 2.0  # z parameter
    unit_size: float = 10.0  # kW per turbine
    capital_cost: float = 2300.0  # $/kW
    om_cost: float = 20.0  # $/kW-year
    replacement_cost: float = 1500.0  # $/kW
    lifetime: int = 20  # years


@dataclass
class NSGABatteryParams:
    """Lead-acid battery parameters (Table 3, Eqs 18-20)."""
    unit_capacity: float = 6.94  # kWh per unit
    charge_efficiency: float = 0.80  # η_bc
    discharge_efficiency: float = 1.00  # η_bf
    self_discharge_rate: float = 0.0002  # 0.02% per hour (σ)
    soc_min: float = 0.20  # 20% minimum SOC
    soc_max: float = 1.00  # 100% maximum SOC
    initial_soc: float = 0.50  # start at 50%
    capital_cost: float = 158.5  # $/kWh (1100$/6.94kWh)
    om_cost: float = 1.44  # $/kWh-year (10$/6.94kWh)
    replacement_cost: float = 144.1  # $/kWh (1000$/6.94kWh)
    lifetime: int = 12  # years


@dataclass
class NSGADieselParams:
    """Diesel generator parameters (Table 3, Eqs 21-22)."""
    unit_size: float = 30.0  # kW per unit
    fuel_intercept: float = 0.246  # F0, L/kWh
    fuel_slope: float = 0.08145  # F1, L/kWh
    fuel_price: float = 0.77  # $/L
    capital_cost: float = 220.0  # $/kW
    om_cost: float = 0.03  # $/hour of operation
    replacement_cost: float = 200.0  # $/kW
    lifetime_hours: int = 15000  # operating hours


@dataclass
class NSGAElectrolyzerParams:
    """PEM Electrolyzer parameters (Eqs 1-4). Costs from paper Table 3."""
    voltage_efficiency: float = 0.70  # η_V
    h2_decomposition_voltage: float = 1.48  # V_H (V)
    faraday_constant: float = 96485.0  # C/mol
    capital_cost: float = 500.0  # $/kW (Table 3)
    om_cost: float = 10.0  # $/kW-year (Table 3)
    replacement_cost: float = 300.0  # $/kW (Table 3)
    lifetime: int = 10  # years (Table 3)

    @property
    def working_voltage(self) -> float:
        """V_elec = V_H / η_V × 100% (Eq 3)."""
        return self.h2_decomposition_voltage / self.voltage_efficiency

    @property
    def h2_rate_mol_per_kwh(self) -> float:
        """Moles of H2 per kWh of electricity (from Eqs 1-2).
        M_elec = I/(2×FC) × 3600, P = I × V_elec
        → M_elec = P/(V_elec × 2 × FC) × 3600
        → rate = 3600 / (2 × FC × V_elec) mol/Wh × 1000 = mol/kWh
        """
        v_elec = self.working_voltage
        return 3600.0 * 1000.0 / (2.0 * self.faraday_constant * v_elec)


@dataclass
class NSGAMethanationParams:
    """Methanation parameters (Eqs 5-6). Costs from paper Table 3.

    Paper model: M_meth(t) = eta_meth × M_elec(t)
    eta_meth is a direct mol H2 → mol CH4 conversion factor.
    Methanation rated power is sized equal to electrolyzer (paper §2.4.2).
    """
    eta_meth: float = 0.80  # Paper Eq 5: 1 mol H2 → 0.80 mol CH4
    ch4_molar_mass: float = 16.0  # g/mol
    capital_cost: float = 400.0  # $/kW (Table 3)
    om_cost: float = 10.0  # $/kW-year (Table 3)
    replacement_cost: float = 200.0  # $/kW (Table 3)
    lifetime: int = 10  # years (Table 3)


@dataclass
class NSGAGasStorageParams:
    """Single-well-vertical (SWV) gas storage (Eqs 7-12). Costs from paper Table 3."""
    pressure: float = 20.0  # MPa (K_SWV)
    temperature: float = 298.0  # K (T_SWV = 25°C)
    gas_constant: float = 8.314e-3  # kPa·m³/(mol·K) for volume calc
    soc_min: float = 0.20  # 20% min SOC
    soc_max: float = 1.00  # 100% max SOC
    dod: float = 0.80  # depth of discharge
    # Cost per m³ of compressed storage volume (Table 3)
    capital_cost: float = 20.0  # $/m³ compressed (Table 3: 20 $/m3)
    om_cost: float = 4.0  # $/m³-year (Table 3)
    replacement_cost: float = 20.0  # $/m³ (assume = capital)
    lifetime: int = 20  # years (Table 3)
    compression_ratio: float = 197.4  # P_storage/P_atm = 20000/101.325

    @property
    def mol_to_m3_stp(self) -> float:
        """Convert mol CH4 to m³ at standard conditions (STP: 1 atm, 298K).
        All gas demand/production tracked in m³ at STP for consistency.
        V = n × R × T / P = n × 8.314e-3 × 298 / 101.325 ≈ n × 0.02447 m³
        """
        return self.temperature * self.gas_constant / 101.325  # kPa at 1 atm


@dataclass
class NSGAROParams:
    """Reverse Osmosis desalination parameters (Eqs 23-26)."""
    specific_energy: float = 4.38  # kWh/m³ (S_DC)
    min_load_fraction: float = 0.25  # 25% minimum load
    capital_cost_per_m3day: float = 532.0  # $/m³/day
    om_cost: float = 0.20  # $/m³ water produced
    chemical_cost: float = 0.06  # $/m³
    membrane_replacements_per_year: int = 2
    membrane_cost_per_m3day: float = 66.5  # $/m³/day (estimated)
    water_tank_cost: float = 255.4  # $/m³ tank capacity
    water_tank_days: float = 2.0  # 2-day storage capacity


@dataclass
class NSGAInverterParams:
    """Inverter parameters (Eqs 27-28). Costs from paper Table 3."""
    efficiency: float = 0.95  # η_inv
    capital_cost: float = 300.0  # $/kW (Table 3)
    om_cost: float = 10.0  # $/kW-year (Table 3)
    lifetime: int = 15  # years (Table 3)
    replacement_cost: float = 300.0  # $/kW (Table 3; was incorrectly 250)


@dataclass
class NSGAEconomicParams:
    """Economic parameters (Eqs 29-44)."""
    discount_rate: float = 0.05  # 5% real discount rate
    inflation_rate: float = 0.02  # 2%
    project_lifetime: int = 25  # years

    def crf(self) -> float:
        """Capital Recovery Factor (Eq 43)."""
        i = self.discount_rate
        n = self.project_lifetime
        return i * (1 + i) ** n / ((1 + i) ** n - 1)

    def present_value_factor(self) -> float:
        """Sum of 1/(1+i)^k for k=1..n, used for O&M and fuel."""
        i = self.discount_rate
        n = self.project_lifetime
        return sum(1.0 / (1 + i) ** k for k in range(1, n + 1))


@dataclass
class NSGACapacityBounds:
    """Optimization bounds for component capacities (NSGA-II decision variables)."""
    pv_min: float = 0.0
    pv_max: float = 1000.0  # kW (paper PV/Batt uses 827 kW)
    wind_min: float = 0.0
    wind_max: float = 500.0  # kW (paper WT/Batt uses 310 kW)
    dg_min: float = 0.0
    dg_max: float = 120.0  # kW (2x paper 60 kW)
    batt_min: float = 0.0
    batt_max: float = 3000.0  # kWh (paper PV/Batt uses 1887 kWh)
    elz_min: float = 0.0
    elz_max: float = 500.0  # kW (paper PV/Batt uses 306 kW)
    gas_storage_min: float = 0.0
    gas_storage_max: float = 1000.0  # m³ compressed (~3x paper 360)


@dataclass
class NSGANSGA2Params:
    """NSGA-II algorithm parameters from paper Table 5."""
    pop_size: int = 500  # Population size (Table 5)
    n_gens: int = 500  # Maximum generations (Table 5)
    p_crossover: float = 0.9  # SBX crossover rate (Table 5)
    p_mutation: float = 0.1  # Polynomial mutation rate (Table 5)
    sbx_eta: float = 15.0  # SBX distribution index
    pm_eta: float = 20.0  # Polynomial mutation distribution index
    seed: int = 42
    lpsp_max: float = 0.01  # Reliability constraint: LPSP <= 1%


@dataclass
class NSGALoadParams:
    """Community load parameters (Table 2, Section 2.2)."""
    summer_daily_kwh: float = 1130.0  # Mar-Oct
    winter_daily_kwh: float = 835.34  # Nov-Feb
    water_demand_m3_day: float = 20.0  # freshwater
    gas_demand_m3_day: float = 330.0  # cooking gas (biogas equivalent)
    population: int = 1000
    households: int = 100


@dataclass
class NSGALocationParams:
    """Saint Martin Island, Bangladesh."""
    latitude: float = 20.633  # °N
    longitude: float = 92.320  # °E
    altitude: float = 3.6  # m above sea level
    timezone: int = 6  # UTC+6 (Bangladesh)
    mean_wind_speed: float = 4.85  # m/s (paper reference)
    mean_solar_kwh_m2_day: float = 4.80  # kWh/m²/day


@dataclass
class NSGASystemParams:
    """Aggregated system parameters."""
    pv: NSGAPVParams = field(default_factory=NSGAPVParams)
    wind: NSGAWindParams = field(default_factory=NSGAWindParams)
    battery: NSGABatteryParams = field(default_factory=NSGABatteryParams)
    diesel: NSGADieselParams = field(default_factory=NSGADieselParams)
    electrolyzer: NSGAElectrolyzerParams = field(default_factory=NSGAElectrolyzerParams)
    methanation: NSGAMethanationParams = field(default_factory=NSGAMethanationParams)
    gas_storage: NSGAGasStorageParams = field(default_factory=NSGAGasStorageParams)
    ro: NSGAROParams = field(default_factory=NSGAROParams)
    inverter: NSGAInverterParams = field(default_factory=NSGAInverterParams)
    economic: NSGAEconomicParams = field(default_factory=NSGAEconomicParams)
    bounds: NSGACapacityBounds = field(default_factory=NSGACapacityBounds)
    nsga2: NSGANSGA2Params = field(default_factory=NSGANSGA2Params)
    load: NSGALoadParams = field(default_factory=NSGALoadParams)
    location: NSGALocationParams = field(default_factory=NSGALocationParams)


DEFAULT_NSGA_PARAMS = NSGASystemParams()
