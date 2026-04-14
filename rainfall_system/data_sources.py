from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests


@dataclass
class OpenMeteoClient:
    base_url: str = "https://api.open-meteo.com/v1/forecast"

    def fetch_hourly(self, lat: float, lon: float, start_date: str, end_date: str) -> dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "precipitation",
                "relative_humidity_2m",
                "dew_point_2m",
                "pressure_msl",
                "temperature_2m",
                "wind_speed_10m",
            ],
            "timezone": "UTC",
        }
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


@dataclass
class SmhiClient:
    # SMHI has multiple open-data endpoints; keep a configurable base.
    base_url: str = "https://opendata-download-metobs.smhi.se/api/version/latest"

    def fetch_latest_observation(self, station_id: int, parameter_id: int, period: str = "latest-day") -> dict[str, Any]:
        url = f"{self.base_url}/parameter/{parameter_id}/station/{station_id}/period/{period}/data.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()


@dataclass
class Era5RequestBuilder:
    dataset: str = "reanalysis-era5-single-levels"

    def build_single_level_request(self, year: int, month: int, day: int, area: list[float]) -> dict[str, Any]:
        mm = f"{month:02d}"
        dd = f"{day:02d}"
        return {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "total_precipitation",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": [str(year)],
            "month": [mm],
            "day": [dd],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,  # [north, west, south, east]
        }


def parse_open_meteo_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
