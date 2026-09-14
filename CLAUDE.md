# CLAUDE.md

## Project Overview
An end-to-end electricity demand (load) forecasting system, built as a portfolio project to demonstrate the full ML lifecycle — from live data ingestion through model deployment. The project simulates a realistic production ML pipeline: raw grid data is ingested on a schedule, cleaned and feature-engineered, used to train and evaluate forecasting models, and the resulting model is served through a containerized API.

**Primary goal:** get hands-on, production-style experience with Databricks, MLflow, ETL pipeline design, and (optionally) AWS SageMaker deployment — filling a data engineering gap alongside existing data science experience.

## Problem Statement
Electric grid operators need accurate short-term load forecasts to balance supply and demand, price electricity, and avoid grid stress. This project predicts electricity demand for a given region ahead of time using historical load, weather, and calendar signals.

## Target Variable
- **What's predicted:** Total electrical load (MW) for a region over a future interval (e.g., next hour or next day / day-ahead).
- **Prediction horizon:** TBD — start with day-ahead (24hr) hourly forecasts, the standard grid-operator framing.
- **Type:** Regression (continuous numeric target).
- **Evaluation metrics:** RMSE, MAE, MAPE — MAPE is the industry-standard metric for load forecasting since it's interpretable as % error.

## Data Sources
- **EIA Open Data API** (`api.eia.gov/v2/electricity/rto/region-data`) — hourly load data, resolved to **ISO New England (respondent code `ISNE`)**, which covers Connecticut. Pulls `type=D` (Demand) only. API key stored in `.env` locally (`API_KEY`) and as a Databricks secret (`energy-demand-forecast/eia_api_key`) for the scheduled job — never committed or synced to the workspace as a file.
- **Weather data** — temperature is the dominant driver of load (heating/cooling degree days). **Open-Meteo** (no API key required): Historical Weather API (ERA5 reanalysis, ~5-day latency) for the one-time backfill; Forecast API (`past_days` param) for live hourly ingestion, since the archive API can't serve "just happened" data. Boston used as a single-point proxy for ISO-NE.
- **Calendar features** — day-of-week, hour-of-day, holidays, solstices/equinoxes — engineered, not pulled from an external source.

## Architecture

```
[ISO API] ---polling job (scheduled)---> [Raw landing zone: Delta table]
                                                    |
                                                    v
                                        [Cleaning + feature engineering]
                                          (lags, rolling avgs, weather
                                           joins, calendar features)
                                                    |
                                                    v
                                          [Feature table: Delta]
                                                    |
                                                    v
                                     [Model training + evaluation]
                                          (tracked in MLflow:
                                           params, metrics, artifacts)
                                                    |
                                                    v
                                        [MLflow Model Registry]
                                                    |
                                                    v
                                   [FastAPI serving layer, Dockerized]
                                                    |
                                                    v
                                        [AWS SageMaker endpoint]
```

**Live data flow:** a scheduled Databricks job polls the ISO API and the weather API hourly, lands new records into Delta tables (demand + weather, separate tasks in the same job), and self-heals against missed runs via a rolling-window MERGE — simulating a real production ingestion cadence rather than a one-time historical pull. Feature engineering, training, and serving currently run against the one-time historical backfill rather than the live tables (see Data Pipeline below).

## Tech Stack
- **Platform:** Databricks (AWS-hosted workspace, connected via the VS Code Databricks extension + Databricks CLI; deployed as a **Databricks Asset Bundle**, `databricks.yml` at repo root). Unity Catalog enabled — tables live under the `workspace` catalog. Workspace only supports **serverless compute** (classic job clusters are rejected), so job tasks omit any cluster/environment spec and default to serverless.
- **Storage:** Delta Lake, Unity Catalog tables `workspace.default.iso_ne_demand`, `workspace.default.boston_weather`, and `workspace.default.iso_ne_features` (engineered feature table).
- **Experiment tracking:** MLflow (Databricks-managed — no separate server needed). Wired up: `train_model.py` logs every run (autolog + custom metrics), registers models to the Unity Catalog registry (`workspace.default.iso_ne_demand_forecast`), and promotes the best candidate via a movable `@prod` alias.
- **Modeling:** compared Linear Regression, ElasticNet, Random Forest, XGBoost, and SVM (scikit-learn + `xgboost`), tuned with Optuna against a chronological train/val split. XGBoost (tuned) is `@prod` — MAPE 7.27%.
- **API:** FastAPI for model serving (`src/serving/app.py`) — `/predict`, `/invocations`, `/ping`.
- **Containerization:** Docker (`src/serving/Dockerfile`), built for `linux/amd64` with legacy Docker V2 manifests (SageMaker rejects the OCI manifests Docker's newer image store produces by default).
- **Deployment:** AWS SageMaker real-time endpoint, bring-your-own-container — image pushed to ECR, model artifact baked into the image (no S3 `ModelDataUrl` needed).
- **Version control:** Git + GitHub (`ShubhanTamhane1/energy-demand-forecast`), bundle deploys via Databricks CLI (`databricks bundle deploy`)

## Data Pipeline (implemented)
- **Historical backfill:** `data/raw/raw_data.ipynb` pulls a full year (2025-08-12 to 2026-08-12) of hourly ISO-NE demand from the EIA API, paginated, saved untouched to `data/raw/iso_ne_demand_2025-08-12_2026-08-12.csv`. This file is the immutable source of truth — never modified in place.
- **EDA:** `notebooks/eda.ipynb` — exploration workbench. Confirmed `respondent`, `respondent-name`, `type`, `type-name`, `value-units` are constant across every row (only `period` and `value` carry information); decided to drop them at the staging step, not from `data/raw`.
- **Scheduled ingestion job:** `src/ingestion/pull_recent_demand.py` (Databricks notebook-format script), deployed as job `pull_recent_demand` via `databricks.yml`.
  - **Trigger:** hourly (`trigger.periodic`, 1 HOUR).
  - **Strategy:** re-pulls a rolling 48-hour lookback window every run and **MERGEs** (upsert on `period`) into `workspace.default.iso_ne_demand` — chosen over plain append-only-latest-hour because it self-heals against missed runs and catches EIA's late revisions to recent readings.
  - **Auth:** EIA key read via `dbutils.secrets.get(scope="energy-demand-forecast", key="eia_api_key")`.
  - Verified working end-to-end (test run confirmed rows landing in the table with the correct timestamp range).
- **Weather backfill + ingestion:** `data/raw/raw_weather_data.ipynb` pulls the matching one-year window from Open-Meteo's Historical Weather API. `src/ingestion/pull_recent_weather.py` runs as a second task in the same hourly job (`pull_recent_demand`), pulling a rolling window from the Forecast API and merging into `workspace.default.boston_weather` — no API key needed. Explicitly filters out any hour later than "now" so only actuals ever land in the table (the Forecast API's `forecast_days` param can't be set below a full day).
- **Feature engineering:** `src/features/build_features.py` — joins demand + weather on `period` (UTC-aligned; demand `period` is unlabeled but confirmed UTC against the known ISO-NE load shape), then adds calendar features (hour/day-of-week/month, weekend/holiday flags via the `holidays` package, cyclical sin/cos encodings), weather features (`did_rain`/`did_snow`, heating/cooling degree hours base 65°F), and demand lag/rolling features (`lag_24h/48h/168h`, trailing `rolling_24h/168h_mean`, leakage-safe via `shift(1)`). Historical output materialized as `workspace.default.iso_ne_features` (uploaded to a UC Volume, then `CREATE TABLE ... AS SELECT` from `read_files`).
- **Model training:** `src/models/train_model.py` (Databricks notebook-format, run as a serverless job submission) — chronological 80/20 split (never random, to avoid future leakage), five model families tuned via Optuna against an inner chronological validation split, logged to MLflow, registered to Unity Catalog.
- **Serving:** `src/serving/download_model.py` pulls the `@prod` artifact locally; `src/serving/app.py` (FastAPI) wraps it; `src/serving/Dockerfile` containerizes; deployed to a SageMaker real-time endpoint (see `architecture-diagram.html` for the full flow).

## Repo Structure
```
energy-demand-forecast/
├── data/                  # gitignored — raw/interim/processed
│   ├── raw/
│   │   ├── raw_data.ipynb                                  # EIA API backfill pull
│   │   ├── iso_ne_demand_2025-08-12_2026-08-12.csv          # 1-year hourly sample, untouched
│   │   ├── raw_weather_data.ipynb                           # Open-Meteo historical backfill pull
│   │   ├── boston_weather_2025-08-12_2026-08-12.csv         # 1-year hourly weather, untouched
│   │   └── pull_recent_weather.ipynb                        # prototype for the live weather job
│   ├── interim/
│   │   └── join_demand_weather.ipynb   # joins demand + weather on period (UTC-aligned)
│   └── processed/
│       └── build_feature_table.ipynb   # runs build_features.py, writes iso_ne_features.csv
├── notebooks/
│   └── eda.ipynb          # exploration only, not production logic
├── src/
│   ├── ingestion/
│   │   ├── pull_recent_demand.py    # hourly Databricks job task: EIA demand -> Delta
│   │   └── pull_recent_weather.py   # hourly Databricks job task: Open-Meteo weather -> Delta
│   ├── features/
│   │   └── build_features.py        # calendar + weather + demand lag/rolling features
│   ├── models/
│   │   └── train_model.py           # MLflow + Optuna training, UC model registry
│   ├── serving/
│   │   ├── app.py                   # FastAPI: /predict, /invocations, /ping
│   │   ├── Dockerfile                # containerizes app.py + model for SageMaker
│   │   ├── serve                     # executable on PATH; SageMaker's `docker run <image> serve`
│   │   ├── download_model.py         # pulls the @prod artifact from the UC registry
│   │   └── model/                    # gitignored — downloaded artifact, rebuild via download_model.py
│   └── __init__.py
├── tests/
├── architecture-diagram.html   # full pipeline architecture diagram
├── databricks.yml         # Databricks Asset Bundle: workspace target + hourly job definition
├── requirements.txt
├── README.md
├── .env                   # gitignored — EIA_API_KEY, ANTHROPIC_API_KEY, DATABRICKS_HOST/TOKEN
└── .gitignore
```

## Development Phases
1. **Setup** — repo skeleton, README, connect Databricks Repos to GitHub ✅
2. **Data access spike** — validate ISO API + weather API access before building anything else ✅ (EIA key validated; Open-Meteo needs no key)
3. **Ingestion pipeline** — historical backfill + scheduled polling job into Delta ✅ (1-year demand + weather backfills in `data/raw/`; hourly Databricks job with two tasks, both verified merging into their Delta tables)
4. **EDA** — explore load patterns, seasonality, weather correlation ✅ (`notebooks/eda.ipynb`: record-type check, constant-column staging decision, daily-average seasonal plot with solstices; weather correlation covered implicitly via feature significance testing in phase 6)
5. **Feature engineering** — lags, rolling stats, calendar/weather joins ✅ (`src/features/build_features.py`; historical output materialized as `workspace.default.iso_ne_features`)
6. **Model training & eval** — baseline model first, iterate, all logged in MLflow ✅ (5 model families, Optuna-tuned, chronological split; OLS significance testing + SHAP used to interpret feature importance; XGBoost (tuned) is best on MAPE)
7. **Model registry** — promote best model via MLflow Model Registry ✅ (`workspace.default.iso_ne_demand_forecast`, `@prod` alias on the winning version)
8. **API + containerization** — FastAPI wrapper, Dockerfile ✅ (`src/serving/app.py`, `src/serving/Dockerfile`, verified locally and in-container)
9. **SageMaker deployment** ✅ (real-time endpoint, BYOC via ECR; verified end-to-end with a live `InvokeEndpoint` call, then torn down to stop billing — `create-endpoint`/`create-model`/`create-endpoint-config` all still in place to recreate on demand)

## Resolved Decisions
- **ISO/region:** ISO New England (`ISNE`), pulled via the EIA Open Data API rather than a direct ISO API — chosen for personal relevance (covers Connecticut) over PJM/ERCOT/CAISO.
- **Compute:** workspace only supports serverless (no classic clusters) — job tasks are defined without an explicit cluster/environment spec.
- **Weather source:** Open-Meteo — Historical Weather API for backfill, Forecast API (`past_days`) for live ingestion, since the historical/archive API has ~5-day latency and can't serve recent actuals.
- **Model:** compared Linear Regression, ElasticNet, Random Forest, XGBoost, SVM — XGBoost (Optuna-tuned) promoted to `@prod` on MAPE (7.27%), though Linear Regression/ElasticNet won on R². The demand signal is strongly linear (dominated by `lag_24h`), which is why linear models were competitive with trees at all.
- **Deployment path:** hand-rolled FastAPI + Dockerfile + SageMaker BYOC, chosen over MLflow's built-in SageMaker auto-deploy — slower, but the point of the project is hands-on production ML experience.
- **SageMaker instance:** `ml.m5.large` (`ml.t2`/`ml.t3` rejected — deprecated/unsupported for real-time endpoints).

## Open Decisions
- Exact prediction horizon (hourly vs. day-ahead)
- Whether to extend into price (LMP) forecasting as a phase-2 model chained off demand predictions
- Whether to port feature engineering + training onto the *live* Delta tables (currently both run against the one-time historical backfill) once the live tables accumulate enough history to be useful
- Whether to pursue a fair rolling one-step-ahead SARIMAX comparison (the one tried was evaluated on an unconditional long-horizon forecast, which isn't the regime this system actually runs in)
