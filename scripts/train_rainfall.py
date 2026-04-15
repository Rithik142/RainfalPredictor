from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rainfall_system.database import init_db
from rainfall_system.features import FEATURE_COLUMNS, make_feature_frame
from rainfall_system.models import persist_models, train_hybrid_models
from rainfall_system.repository import save_training_run


def time_aware_split(
    df: pd.DataFrame,
    time_col: str = "time",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    n = len(sorted_df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return sorted_df.iloc[:i_train], sorted_df.iloc[i_train:i_val], sorted_df.iloc[i_val:]


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_rain_24h"] = (
        out["precipitation"].rolling(24, min_periods=1).sum().shift(-23).fillna(0.0)
    )
    return out


def find_best_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (probs >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def evaluate(
    val: pd.DataFrame,
    test: pd.DataFrame,
    models,
    extreme_threshold_mm: float = 10.0,
) -> dict[str, float]:
    x_val = val[FEATURE_COLUMNS]
    y_val_amount = val["target_rain_24h"]
    y_val_occ = (y_val_amount > 0.1).astype(int)
    y_val_ext = (y_val_amount > extreme_threshold_mm).astype(int)

    occ_val_prob = models.occurrence.predict_proba(x_val)[:, 1]
    ext_val_prob = models.extreme.predict_proba(x_val)[:, 1]

    occ_cut = find_best_threshold(y_val_occ, occ_val_prob)
    ext_cut = find_best_threshold(y_val_ext, ext_val_prob) if y_val_ext.nunique() > 1 else 0.5

    x_test = test[FEATURE_COLUMNS]
    y_amount = test["target_rain_24h"]
    y_occ = (y_amount > 0.1).astype(int)
    y_ext = (y_amount > extreme_threshold_mm).astype(int)

    occ_prob = models.occurrence.predict_proba(x_test)[:, 1]
    occ_pred = (occ_prob >= occ_cut).astype(int)

    amt_pred = models.quantile_p50.predict(x_test)

    ext_prob = models.extreme.predict_proba(x_test)[:, 1]
    ext_pred = (ext_prob >= ext_cut).astype(int)

    return {
        "occ_threshold": occ_cut,
        "ext_threshold": ext_cut,
        "occ_f1": float(f1_score(y_occ, occ_pred, zero_division=0)),
        "occ_auc": float(roc_auc_score(y_occ, occ_prob)) if y_occ.nunique() > 1 else 0.5,
        "amount_mae": float(mean_absolute_error(y_amount, amt_pred)),
        "amount_rmse": float(mean_squared_error(y_amount, amt_pred) ** 0.5),
        "ext_precision": float(precision_score(y_ext, ext_pred, zero_division=0)),
        "ext_recall": float(recall_score(y_ext, ext_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rainfall hybrid models")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to hourly weather CSV with at least time + precipitation + meteorology columns",
    )
    args = parser.parse_args()

    init_db()

    raw = pd.read_csv(args.input_csv)
    raw = add_targets(raw)
    feats = make_feature_frame(raw)
    dataset = pd.concat([raw.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    train, val, test = time_aware_split(dataset)

    models = train_hybrid_models(train)
    metrics = evaluate(val, test, models)
    persist_models(models)

    run_id = save_training_run(
        model_version="rain_hybrid_v3.0",
        train_rows=len(train),
        validation_rows=len(val),
        test_rows=len(test),
        metrics=metrics,
    )
    print(f"Saved models to artifacts/ (training_run_id={run_id})")
    print(metrics)


if __name__ == "__main__":
    main()