from datetime import date

from pydantic import BaseModel, Field


class RainfallPredictRequest(BaseModel):
    region: str = Field(..., examples=["Stockholm"])
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    horizon_hours: int = Field(default=72, ge=1, le=240)
    start_date: date


class RainfallPredictResponse(BaseModel):
    prediction_id: int
    rain_probability_24h: float
    rainfall_mm_p10_24h: float
    rainfall_mm_p50_24h: float
    rainfall_mm_p90_24h: float
    extreme_rain_risk: float
    extreme_threshold_mm: float
    model_version: str
