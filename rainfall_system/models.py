from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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

    def _align_features_for_model(self, model: Any, features: pd.DataFrame) -> pd.DataFrame:
        expected = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
        return features.reindex(columns=expected, fill_value=0.0).tail(1)

    def predict_from_features(self, features: pd.DataFrame) -> dict[str, float]:
        x_occ = self._align_features_for_model(self._models.occurrence, features)
        x_q10 = self._align_features_for_model(self._models.quantile_p10, features)
        x_q50 = self._align_features_for_model(self._models.quantile_p50, features)
        x_q90 = self._align_features_for_model(self._models.quantile_p90, features)
        x_ext = self._align_features_for_model(self._models.extreme, features)

        rain_prob = float(self._models.occurrence.predict_proba(x_occ)[0, 1])
        q10 = max(0.0, float(self._models.quantile_p10.predict(x_q10)[0]))
        q50 = max(0.0, float(self._models.quantile_p50.predict(x_q50)[0]))
        q90 = max(q50, float(self._models.quantile_p90.predict(x_q90)[0]))
        extreme_prob = float(self._models.extreme.predict_proba(x_ext)[0, 1])

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


def train_hybrid_models(
    df: pd.DataFrame,
    rain_threshold_mm: float = 0.1,
    extreme_threshold_mm: float = 10.0,
) -> TrainedRainfallModels:
    x = df[FEATURE_COLUMNS]
    y_occurrence = (df["target_rain_24h"] > rain_threshold_mm).astype(int)
    y_amount = df["target_rain_24h"].clip(lower=0.0)
    y_extreme = (df["target_rain_24h"] > extreme_threshold_mm).astype(int)

    occurrence = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    occurrence.fit(x, y_occurrence)

    quantile_p10 = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.1,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    quantile_p50 = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.5,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    quantile_p90 = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.9,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    quantile_p10.fit(x, y_amount)
    quantile_p50.fit(x, y_amount)
    quantile_p90.fit(x, y_amount)

    extreme = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.85,
        random_state=42,
    )
    extreme.fit(x, y_extreme)

    return TrainedRainfallModels(
        occurrence=occurrence,
        quantile_p10=quantile_p10,
        quantile_p50=quantile_p50,
        quantile_p90=quantile_p90,
        extreme=extreme,
    )


def persist_models(models: TrainedRainfallModels, paths: Paths | None = None) -> None:
    target = paths or Paths()
    target.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.occurrence, target.classifier_path)
    joblib.dump(models.quantile_p10, target.quantile_p10_path)
    joblib.dump(models.quantile_p50, target.quantile_p50_path)
    joblib.dump(models.quantile_p90, target.quantile_p90_path)
    joblib.dump(models.extreme, target.extreme_path)