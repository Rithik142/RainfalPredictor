from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "precip_1h",
    "precip_3h",
    "precip_6h",
    "precip_24h",
    "rh_2m",
    "dewpoint_depression",
    "pressure_tendency_3h",
    "wind_speed_10m",
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
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out


def add_rain_lag_features(df: pd.DataFrame, precip_col: str = "precipitation") -> pd.DataFrame:
    out = df.copy()
    out["precip_1h"] = out[precip_col].rolling(1, min_periods=1).sum()
    out["precip_3h"] = out[precip_col].rolling(3, min_periods=1).sum()
    out["precip_6h"] = out[precip_col].rolling(6, min_periods=1).sum()
    out["precip_24h"] = out[precip_col].rolling(24, min_periods=1).sum()
    return out


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dewpoint_depression"] = out["temperature_2m"] - out["dew_point_2m"]
    out["pressure_tendency_3h"] = out["pressure_msl"].diff(3).fillna(0.0)
    out["rh_2m"] = out["relative_humidity_2m"]
    return out


def add_antecedent_wetness_index(df: pd.DataFrame, precip_col: str = "precipitation", decay: float = 0.85) -> pd.DataFrame:
    out = df.copy()
    awi_values = []
    running = 0.0
    for p in out[precip_col].fillna(0.0).to_numpy():
        running = decay * running + p
        awi_values.append(running)
    out["awi"] = awi_values
    return out


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_rain_lag_features(out)
    out = add_physical_features(out)
    out = add_antecedent_wetness_index(out)
    return out[FEATURE_COLUMNS].fillna(0.0)
