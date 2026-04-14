# Rainfall Prediction System (Competition-Ready)

This repository contains a concrete implementation of an **advanced rainfall prediction backend** with three outputs:

1. **Rain occurrence** in the next 24h (classification)
2. **Rain amount** in mm (quantile regression: P10/P50/P90)
3. **Extreme rain risk** (probability of threshold exceedance, default 20 mm/day)

It is designed to be API-first and easy to integrate into a groundwater forecasting stack.

## Backend + Database

The backend is implemented in **FastAPI** (`rainfall_system/api.py`).

Database is configured through `DATABASE_URL` and uses:

- **SQLite by default**: `sqlite:///./rainfall.db`
- **PostgreSQL (recommended for production)** via env var, e.g.
  `postgresql+psycopg://user:pass@host:5432/rainfall`

The app persists:

- `rain_predictions` table: request + predicted outputs for traceability
- `training_runs` table: model metrics and dataset split sizes per run

## Clone/Download to Your Computer

### Option A: Git (recommended)

```bash
git clone <YOUR_REPO_URL>
cd RainfalPredictor
```

### Option B: Download ZIP

1. Open your Git hosting page.
2. Click **Code** -> **Download ZIP**.
3. Extract the folder.
4. Open terminal in the extracted folder.

## Setup (Mac/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn rainfall_system.api:app --reload
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn rainfall_system.api:app --reload
```

API will run at: `http://127.0.0.1:8000`

## Quick API Test

Use Swagger UI: `http://127.0.0.1:8000/docs`

Or with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict_rainfall \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Stockholm",
    "lat": 59.3293,
    "lon": 18.0686,
    "horizon_hours": 72,
    "start_date": "2026-04-15"
  }'
```

## Train the AI Model

Prepare an hourly CSV with at least these columns:

- `time`
- `precipitation`
- `relative_humidity_2m`
- `dew_point_2m`
- `pressure_msl`
- `temperature_2m`
- `wind_speed_10m`

Then train:

```bash
python scripts/train_rainfall.py --input-csv data/weather_hourly.csv
```

Training will:

- build features + 24h rainfall target,
- do time-aware split (train/val/test),
- train staged models,
- persist models to `artifacts/`,
- store metrics in `training_runs` database table.

## Android Studio Integration

You can connect your Android app directly to this backend.

- Full guide (Retrofit + Kotlin models + emulator networking):
  `android/README_ANDROID.md`

Most important Android networking note:

- Android emulator must call local backend via `http://10.0.2.2:8000/` (not `localhost`).

## Architecture

- `rainfall_system/data_sources.py`
  - SMHI observation fetching
  - ERA5 request payload builder
  - Open-Meteo forecast fetching
- `rainfall_system/features.py`
  - Lag/rolling precipitation features
  - Humidity and dew-point proxies
  - Pressure tendency
  - Seasonal cyclic encodings
  - Antecedent wetness index (AWI)
- `rainfall_system/models.py`
  - Stage A: rain/no-rain classifier
  - Stage B: quantile rainfall models (P10/P50/P90)
  - Stage C: extreme rain classifier + calibration support
- `rainfall_system/api.py`
  - `POST /predict_rainfall` and `/health`
- `scripts/train_rainfall.py`
  - Time-aware split training and metrics logging

## Validation Guidance

Use a strictly time-aware split:

- Train: oldest years
- Validation: recent period
- Test: latest holdout period

Recommended metrics:

- Occurrence: F1, ROC-AUC, Brier score
- Amount: MAE/RMSE, quantile loss
- Extremes: precision/recall on exceedance days, precision@k
- Reliability/calibration plots for probabilities
