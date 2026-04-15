from __future__ import annotations

from datetime import timedelta

import pandas as pd
from fastapi import FastAPI

from rainfall_system.data_sources import OpenMeteoClient
from rainfall_system.database import init_db
from rainfall_system.features import make_feature_frame
from rainfall_system.models import RainfallModelService
from rainfall_system.repository import save_prediction
from rainfall_system.schemas import RainfallPredictRequest, RainfallPredictResponse

app = FastAPI(title="Rainfall Prediction API", version="1.1.0")

open_meteo = OpenMeteoClient()
model_service = RainfallModelService()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_rainfall", response_model=RainfallPredictResponse)
def predict_rainfall(payload: RainfallPredictRequest) -> RainfallPredictResponse:
    end_date = payload.start_date + timedelta(days=max(1, payload.horizon_hours // 24))
    raw = open_meteo.fetch_hourly(
        lat=payload.lat,
        lon=payload.lon,
        start_date=payload.start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    hourly = raw.get("hourly", {})
    df = pd.DataFrame(
        {
            "time": hourly.get("time", []),
            "precipitation": hourly.get("precipitation", []),
            "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
            "dew_point_2m": hourly.get("dew_point_2m", []),
            "pressure_msl": hourly.get("pressure_msl", []),
            "temperature_2m": hourly.get("temperature_2m", []),
            "wind_speed_10m": hourly.get("wind_speed_10m", []),
        }
    )

    result: dict[str, float | str]
    if df.empty:
        result = {
            "rain_probability_24h": 0.0,
            "rainfall_mm_p10_24h": 0.0,
            "rainfall_mm_p50_24h": 0.0,
            "rainfall_mm_p90_24h": 0.0,
            "extreme_rain_risk": 0.0,
            "extreme_threshold_mm": 20.0,
            "model_version": "rain_hybrid_v1.0",
        }
    else:
        features = make_feature_frame(df)
        result = model_service.predict_from_features(features)

    prediction_id = save_prediction(
        payload={
            "region": payload.region,
            "lat": payload.lat,
            "lon": payload.lon,
            "horizon_hours": payload.horizon_hours,
            "start_date": payload.start_date.isoformat(),
        },
        prediction=result,
    )

    return RainfallPredictResponse(prediction_id=prediction_id, **result)
