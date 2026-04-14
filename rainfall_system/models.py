from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from rainfall_system.config import ModelConfig, Paths
from rainfall_system.features import FEATURE_COLUMNS


@dataclass
class TrainedRainfallModels:
    occurrence: Any
    quantile_p10: Any
    quantile_p50: Any
    quantile_p90: Any
    extreme: Any


class RainfallModelService:
    def __init__(self, paths: Paths | None = None, config: ModelConfig | None = None) -> None:
        self.paths = paths or Paths()
        self.config = config or ModelConfig()
        self._models = self._load_models()

    def _load_models(self) -> TrainedRainfallModels:
        return TrainedRainfallModels(
            occurrence=_load_or_default_classifier(self.paths.classifier_path),
            quantile_p10=_load_or_default_quantile(self.paths.quantile_p10_path, alpha=0.1),
            quantile_p50=_load_or_default_quantile(self.paths.quantile_p50_path, alpha=0.5),
            quantile_p90=_load_or_default_quantile(self.paths.quantile_p90_path, alpha=0.9),
            extreme=_load_or_default_classifier(self.paths.extreme_path),
        )

    def predict_from_features(self, features: pd.DataFrame) -> dict[str, float]:
        x = features[FEATURE_COLUMNS].tail(1)
        rain_prob = float(self._models.occurrence.predict_proba(x)[0, 1])
        q10 = max(0.0, float(self._models.quantile_p10.predict(x)[0]))
        q50 = max(0.0, float(self._models.quantile_p50.predict(x)[0]))
        q90 = max(q50, float(self._models.quantile_p90.predict(x)[0]))
        extreme_prob = float(self._models.extreme.predict_proba(x)[0, 1])
        return {
            "rain_probability_24h": round(rain_prob, 4),
            "rainfall_mm_p10_24h": round(q10, 3),
            "rainfall_mm_p50_24h": round(q50, 3),
            "rainfall_mm_p90_24h": round(q90, 3),
            "extreme_rain_risk": round(extreme_prob, 4),
            "extreme_threshold_mm": self.config.extreme_threshold_mm,
            "model_version": self.config.model_version,
        }


def _load_or_default_classifier(path: Path):
    if path.exists():
        return joblib.load(path)
    model = LogisticRegression(max_iter=500)
    x = np.random.randn(64, len(FEATURE_COLUMNS))
    y = (np.random.rand(64) > 0.65).astype(int)
    model.fit(x, y)
    return model


def _load_or_default_quantile(path: Path, alpha: float):
    if path.exists():
        return joblib.load(path)
    model = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)
    x = np.random.randn(128, len(FEATURE_COLUMNS))
    y = np.abs(np.random.randn(128) * 5.0)
    model.fit(x, y)
    return model


def train_hybrid_models(df: pd.DataFrame, rain_threshold_mm: float = 0.1, extreme_threshold_mm: float = 20.0) -> TrainedRainfallModels:
    x = df[FEATURE_COLUMNS]
    y_occurrence = (df["target_rain_24h"] > rain_threshold_mm).astype(int)
    y_amount = df["target_rain_24h"].clip(lower=0.0)
    y_extreme = (df["target_rain_24h"] > extreme_threshold_mm).astype(int)

    occ = GradientBoostingClassifier(random_state=42)
    occ.fit(x, y_occurrence)

    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=42)
    q50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42)
    q10.fit(x, y_amount)
    q50.fit(x, y_amount)
    q90.fit(x, y_amount)

    ext_base = GradientBoostingClassifier(random_state=42)
    ext_cal = CalibratedClassifierCV(ext_base, method="isotonic", cv=3)
    ext_cal.fit(x, y_extreme)

    return TrainedRainfallModels(occurrence=occ, quantile_p10=q10, quantile_p50=q50, quantile_p90=q90, extreme=ext_cal)


def persist_models(models: TrainedRainfallModels, paths: Paths | None = None) -> None:
    target = paths or Paths()
    target.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.occurrence, target.classifier_path)
    joblib.dump(models.quantile_p10, target.quantile_p10_path)
    joblib.dump(models.quantile_p50, target.quantile_p50_path)
    joblib.dump(models.quantile_p90, target.quantile_p90_path)
    joblib.dump(models.extreme, target.extreme_path)
