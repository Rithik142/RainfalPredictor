from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lat",
    "lon",
    "precip_1h",
    "precip_3h",
    "precip_6h",
    "precip_12h",
    "precip_24h",
    "precip_48h",
    "precip_72h",
    "rh_2m",
    "dewpoint_depression",
    "temp_change_3h",
    "temp_change_6h",
    "pressure_tendency_3h",
    "pressure_tendency_6h",
    "wind_speed_10m",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "awi",
]


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out["month"] = out[time_col].dt.month
    out["day_of_year"] = out[time_col].dt.dayofyear
    out["hour"] = out[time_col].dt.hour

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_rain_lag_features(df: pd.DataFrame, precip_col: str = "precipitation") -> pd.DataFrame:
    out = df.copy()
    out["precip_1h"] = out[precip_col].rolling(1, min_periods=1).sum()
    out["precip_3h"] = out[precip_col].rolling(3, min_periods=1).sum()
    out["precip_6h"] = out[precip_col].rolling(6, min_periods=1).sum()
    out["precip_12h"] = out[precip_col].rolling(12, min_periods=1).sum()
    out["precip_24h"] = out[precip_col].rolling(24, min_periods=1).sum()
    out["precip_48h"] = out[precip_col].rolling(48, min_periods=1).sum()
    out["precip_72h"] = out[precip_col].rolling(72, min_periods=1).sum()
    return out


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dewpoint_depression"] = out["temperature_2m"] - out["dew_point_2m"]
    out["pressure_tendency_3h"] = out["pressure_msl"].diff(3).fillna(0.0)
    out["pressure_tendency_6h"] = out["pressure_msl"].diff(6).fillna(0.0)
    out["temp_change_3h"] = out["temperature_2m"].diff(3).fillna(0.0)
    out["temp_change_6h"] = out["temperature_2m"].diff(6).fillna(0.0)
    out["rh_2m"] = out["relative_humidity_2m"]
    return out


def add_antecedent_wetness_index(
    df: pd.DataFrame,
    precip_col: str = "precipitation",
    decay: float = 0.92,
) -> pd.DataFrame:
    out = df.copy()
    awi_values = []
    running = 0.0
    for p in out[precip_col].fillna(0.0).to_numpy():
        running = decay * running + p
        awi_values.append(running)
    out["awi"] = awi_values
    return out


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "lat" not in out.columns:
        out["lat"] = 0.0
    if "lon" not in out.columns:
        out["lon"] = 0.0

    out = add_time_features(out)
    out = add_rain_lag_features(out)
    out = add_physical_features(out)
    out = add_antecedent_wetness_index(out)

    return out[FEATURE_COLUMNS].fillna(0.0)