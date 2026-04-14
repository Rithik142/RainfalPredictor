from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_version: str = "rain_hybrid_v1.0"
    extreme_threshold_mm: float = 20.0
    horizon_hours_default: int = 72


@dataclass(frozen=True)
class Paths:
    artifacts_dir: Path = Path("artifacts")
    classifier_path: Path = Path("artifacts/rain_occurrence.pkl")
    quantile_p10_path: Path = Path("artifacts/rain_amount_q10.pkl")
    quantile_p50_path: Path = Path("artifacts/rain_amount_q50.pkl")
    quantile_p90_path: Path = Path("artifacts/rain_amount_q90.pkl")
    extreme_path: Path = Path("artifacts/extreme_risk.pkl")
