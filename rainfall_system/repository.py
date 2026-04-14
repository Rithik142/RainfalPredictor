from __future__ import annotations

from typing import Any

from rainfall_system.database import RainPredictionRecord, TrainingRunRecord, session_scope


def save_prediction(payload: dict[str, Any], prediction: dict[str, Any]) -> int:
    with session_scope() as session:
        record = RainPredictionRecord(
            region=payload["region"],
            lat=payload["lat"],
            lon=payload["lon"],
            horizon_hours=payload["horizon_hours"],
            start_date=payload["start_date"],
            prediction=prediction,
        )
        session.add(record)
        session.flush()
        return int(record.id)


def save_training_run(model_version: str, train_rows: int, validation_rows: int, test_rows: int, metrics: dict[str, float]) -> int:
    with session_scope() as session:
        record = TrainingRunRecord(
            model_version=model_version,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            metrics=metrics,
        )
        session.add(record)
        session.flush()
        return int(record.id)
