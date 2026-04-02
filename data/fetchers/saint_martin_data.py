"""Data provider for Saint Martin Island, Bangladesh.

Fetches NASA POWER hourly data and generates load profiles
matching Ahmed et al. (2024) paper specifications.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from config.nsga_parameters import NSGALocationParams, NSGALoadParams, DEFAULT_NSGA_PARAMS


class SaintMartinDataProvider:
    """Provides meteorological and load data for Saint Martin Island."""

    def __init__(self, params: NSGALocationParams = None, load_params: NSGALoadParams = None):
        self.loc = params or DEFAULT_NSGA_PARAMS.location
        self.load = load_params or DEFAULT_NSGA_PARAMS.load

    def fetch_meteorological_data(self, year: int = 2022) -> pd.DataFrame:
        """Fetch hourly met data from NASA POWER or generate synthetic.

        Returns DataFrame with columns: irradiance (W/m²), temperature (°C),
        wind_speed (m/s at 10m reference height)
        """
        from data.fetchers.nasa_power import NASAPowerClient
        client = NASAPowerClient(
            latitude=self.loc.latitude,
            longitude=self.loc.longitude,
        )
        try:
            df = client.fetch_year(year, use_cache=True)
            # Use wind at 10m (paper's reference height is 10m, hub is 20m)
            if "wind_speed_10m" in df.columns:
                df["wind_speed"] = df["wind_speed_10m"]
            elif "wind_speed_50m" in df.columns:
                # Scale 50m back to 10m
                df["wind_speed"] = df["wind_speed_50m"] * (10.0 / 50.0) ** 0.14
            return df
        except Exception as e:
            print(f"NASA POWER fetch failed: {e}")
            print("Generating synthetic data for Saint Martin...")
            return self._generate_synthetic(year)

    def _generate_synthetic(self, year: int) -> pd.DataFrame:
        """Generate synthetic met data matching paper's reference values.

        Paper: mean wind = 4.85 m/s, mean solar = 4.80 kWh/m²/day
        """
        np.random.seed(42 + year)
        dates = pd.date_range(start=f"{year}-01-01", periods=8760, freq="h")
        hours = np.arange(8760) % 24
        day_of_year = np.arange(8760) // 24

        # --- Solar irradiance ---
        # 4.80 kWh/m²/day ≈ 200 W/m² 24h-avg, peak ~1000 W/m²
        solar_pattern = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        # Monsoon: cloudier Jun-Sep, clearer Nov-Feb
        seasonal = 1.0 + 0.15 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        cloud_factor = 0.75 + 0.5 * np.random.random(8760)
        irradiance = solar_pattern * 1000.0 * seasonal * cloud_factor
        irradiance = np.clip(irradiance, 0, 1100)

        # --- Wind speed at 10m ---
        # Paper mean 4.85 m/s, peak in monsoon (Jun-Sep)
        monsoon_boost = 1.0 + 0.3 * np.where(
            (day_of_year >= 150) & (day_of_year <= 270), 1.0, 0.0
        )
        diurnal = 1.0 + 0.2 * np.sin((hours - 14) * np.pi / 12)
        wind_base = 4.85 * diurnal * monsoon_boost
        wind_speed = wind_base * (0.6 + 0.8 * np.random.random(8760))
        wind_speed = np.clip(wind_speed, 0.3, 18.0)

        # --- Temperature ---
        # Tropical: 22-34°C, warmer Apr-Jun
        temp_base = 26.0 + 4.0 * np.sin((hours - 14) * np.pi / 12)
        temp_seasonal = 2.0 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        temperature = temp_base + temp_seasonal + np.random.normal(0, 1, 8760)
        temperature = np.clip(temperature, 20, 38)

        df = pd.DataFrame({
            "irradiance": irradiance,
            "temperature": temperature,
            "wind_speed": wind_speed,
        }, index=dates)

        return df

    def generate_electrical_load(self, year: int = 2022) -> np.ndarray:
        """Generate hourly electrical demand profile (kW).

        Summer (Mar-Oct): 1130 kWh/day = 47.08 kW avg
        Winter (Nov-Feb): 835.34 kWh/day = 34.81 kW avg
        """
        dates = pd.date_range(start=f"{year}-01-01", periods=8760, freq="h")
        hours = np.arange(8760) % 24
        months = dates.month

        # Base load by season
        is_summer = (months >= 3) & (months <= 10)
        daily_kwh = np.where(is_summer, self.load.summer_daily_kwh, self.load.winter_daily_kwh)

        # Diurnal pattern: morning peak 8-11, evening peak 18-22
        hourly_weight = np.ones(24)
        hourly_weight[0:6] = 0.5  # night low
        hourly_weight[6:8] = 0.8  # early morning
        hourly_weight[8:11] = 1.5  # morning peak
        hourly_weight[11:14] = 1.0  # midday
        hourly_weight[14:18] = 0.9  # afternoon
        hourly_weight[18:22] = 1.6  # evening peak
        hourly_weight[22:24] = 0.7  # late night
        hourly_weight = hourly_weight / hourly_weight.sum() * 24  # normalize to sum=24

        # Apply pattern
        load_kw = np.zeros(8760)
        for h_idx in range(8760):
            hour = hours[h_idx]
            load_kw[h_idx] = (daily_kwh[h_idx] / 24.0) * hourly_weight[hour]

        # Add some noise (±5%)
        np.random.seed(123 + year)
        noise = 1.0 + 0.05 * np.random.randn(8760)
        load_kw = load_kw * noise
        load_kw = np.maximum(load_kw, 1.0)

        return load_kw

    def generate_water_demand(self, year: int = 2022) -> np.ndarray:
        """Generate hourly water demand profile (m³/h).

        20 m³/day = constant 0.833 m³/h
        """
        return np.full(8760, self.load.water_demand_m3_day / 24.0)

    def generate_gas_demand(self, year: int = 2022) -> np.ndarray:
        """Generate hourly cooking gas demand profile (m³/h).

        330 m³/day with 3 meal peaks (7-8am, 12-1pm, 7-8pm).
        """
        hours = np.arange(8760) % 24
        hourly_weight = np.ones(24) * 0.2  # base (pilot lights, etc.)
        # Breakfast: 6-9 AM
        hourly_weight[6:9] = 2.5
        # Lunch: 11 AM - 2 PM
        hourly_weight[11:14] = 2.0
        # Dinner: 6-9 PM
        hourly_weight[18:21] = 3.0
        hourly_weight = hourly_weight / hourly_weight.sum() * 24.0

        gas_m3_h = np.zeros(8760)
        for h_idx in range(8760):
            hour = hours[h_idx]
            gas_m3_h[h_idx] = (self.load.gas_demand_m3_day / 24.0) * hourly_weight[hour]

        return gas_m3_h

    def load_all_data(self, year: int = 2022) -> dict:
        """Load all data needed for optimization.

        Returns dict with:
        - met_data: DataFrame with irradiance, temperature, wind_speed
        - elec_demand: hourly electrical demand (kW), shape (8760,)
        - water_demand: hourly water demand (m³/h), shape (8760,)
        - gas_demand: hourly gas demand (m³/h), shape (8760,)
        - irradiance_factor: normalized PV capacity factor (0-1)
        - wind_factor: normalized wind capacity factor (0-1)
        """
        met_data = self.fetch_meteorological_data(year)
        elec_demand = self.generate_electrical_load(year)
        water_demand = self.generate_water_demand(year)
        gas_demand = self.generate_gas_demand(year)

        # PV capacity factor with derating + temp correction (Eq 13-14)
        pv = DEFAULT_NSGA_PARAMS.pv
        irr = met_data["irradiance"].values
        temp = met_data["temperature"].values

        t_cell = temp + irr * ((pv.noct - 20.0) / 800.0) * (1.0 - pv.efficiency / pv.transmittance_absorptance)
        delta_t = t_cell - 25.0
        temp_factor = 1.0 + pv.temp_coefficient * delta_t
        irradiance_factor = pv.derating_factor * (irr / 1000.0) * temp_factor
        irradiance_factor = np.clip(irradiance_factor, 0, 1)

        # Wind capacity factor (Eq 15-17)
        wp = DEFAULT_NSGA_PARAMS.wind
        ws = met_data["wind_speed"].values
        # Adjust from reference height to hub height
        ws_hub = ws * (wp.hub_height / wp.reference_height) ** wp.wind_shear_exponent

        z = wp.weibull_shape
        wind_factor = np.zeros_like(ws_hub)
        mask_op = (ws_hub >= wp.v_cut_in) & (ws_hub <= wp.v_rated)
        mask_rated = (ws_hub > wp.v_rated) & (ws_hub <= wp.v_cut_off)
        # P = a + b × V^z where a = P_rated × V_ci^z / (V_ci^z - V_r^z)
        # and b = P_rated / (V_r^z - V_ci^z)
        # Simplified: factor = (V^z - V_ci^z) / (V_r^z - V_ci^z)
        denom = wp.v_rated ** z - wp.v_cut_in ** z
        wind_factor[mask_op] = (ws_hub[mask_op] ** z - wp.v_cut_in ** z) / denom
        wind_factor[mask_rated] = 1.0

        return {
            "met_data": met_data,
            "elec_demand": elec_demand,
            "water_demand": water_demand,
            "gas_demand": gas_demand,
            "irradiance_factor": irradiance_factor,
            "wind_factor": wind_factor,
        }
