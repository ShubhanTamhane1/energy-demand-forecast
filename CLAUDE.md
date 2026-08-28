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
- **Weather data** — temperature is the dominant driver of load (heating/cooling degree days). Source TBD (e.g., Open-Meteo, no API key required).
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
                                  [Optional: AWS SageMaker endpoint]
```

**Live data flow:** a scheduled Databricks job polls the ISO API at a regular interval (e.g., hourly), lands new records into a Delta table, and re-triggers the feature pipeline — simulating a real production ingestion cadence rather than a one-time historical pull.

## Tech Stack
- **Platform:** Databricks (AWS-hosted workspace, connected via the VS Code Databricks extension + Databricks CLI; deployed as a **Databricks Asset Bundle**, `databricks.yml` at repo root). Unity Catalog enabled — tables live under the `workspace` catalog. Workspace only supports **serverless compute** (classic job clusters are rejected), so job tasks omit any cluster/environment spec and default to serverless.
- **Storage:** Delta Lake, Unity Catalog table `workspace.default.iso_ne_demand` (columns: `period` TIMESTAMP, `value` DOUBLE — constant metadata columns dropped at staging, see Data Pipeline below).
- **Experiment tracking:** MLflow (Databricks-managed — no separate server needed; not yet wired up)
- **Modeling:** traditional ML (e.g., gradient boosting / regression models) via scikit-learn or similar — no deep learning needed for this problem
- **API:** FastAPI for model serving
- **Containerization:** Docker
- **Deployment (stretch goal):** AWS SageMaker endpoint
- **Version control:** Git + GitHub (`ShubhanTamhane1/energy-demand-forecast`), bundle deploys via Databricks CLI (`databricks bundle deploy`)

## Data Pipeline (implemented)
- **Historical backfill:** `data/raw/raw_data.ipynb` pulls a full year (2025-08-12 to 2026-08-12) of hourly ISO-NE demand from the EIA API, paginated, saved untouched to `data/raw/iso_ne_demand_2025-08-12_2026-08-12.csv`. This file is the immutable source of truth — never modified in place.
- **EDA:** `notebooks/eda.ipynb` — exploration workbench. Confirmed `respondent`, `respondent-name`, `type`, `type-name`, `value-units` are constant across every row (only `period` and `value` carry information); decided to drop them at the staging step, not from `data/raw`.
- **Scheduled ingestion job:** `src/ingestion/pull_recent_demand.py` (Databricks notebook-format script), deployed as job `pull_recent_demand` via `databricks.yml`.
  - **Trigger:** hourly (`trigger.periodic`, 1 HOUR).
  - **Strategy:** re-pulls a rolling 48-hour lookback window every run and **MERGEs** (upsert on `period`) into `workspace.default.iso_ne_demand` — chosen over plain append-only-latest-hour because it self-heals against missed runs and catches EIA's late revisions to recent readings.
  - **Auth:** EIA key read via `dbutils.secrets.get(scope="energy-demand-forecast", key="eia_api_key")`.
  - Verified working end-to-end (test run confirmed rows landing in the table with the correct timestamp range).

## Repo Structure
```
energy-demand-forecast/
├── data/                  # gitignored — raw/interim/processed
│   └── raw/
│       ├── raw_data.ipynb                                  # EIA API backfill pull
│       └── iso_ne_demand_2025-08-12_2026-08-12.csv          # 1-year hourly sample, untouched
├── notebooks/
│   └── eda.ipynb          # exploration only, not production logic
├── src/
│   ├── ingestion/
│   │   └── pull_recent_demand.py   # hourly Databricks job: pulls + merges into Delta
│   ├── features/          # cleaning + feature engineering functions (not yet built)
│   ├── models/             # training/eval scripts (not yet built)
│   └── serving/            # FastAPI app for model serving (not yet built)
├── tests/
├── mlflow/                # experiment configs if needed
├── databricks.yml         # Databricks Asset Bundle: workspace target + hourly job definition
├── requirements.txt
├── README.md
├── .env                   # gitignored — EIA_API_KEY, ANTHROPIC_API_KEY, DATABRICKS_HOST/TOKEN
└── .gitignore
```

## Development Phases
1. **Setup** — repo skeleton, README, connect Databricks Repos to GitHub ✅
2. **Data access spike** — validate ISO API + weather API access before building anything else ✅ (EIA key validated; weather API still TBD)
3. **Ingestion pipeline** — historical backfill + scheduled polling job into Delta ✅ (1-year backfill in `data/raw/`; hourly Databricks job deployed and verified, merges into `workspace.default.iso_ne_demand`)
4. **EDA** — explore load patterns, seasonality, weather correlation 🔶 in progress (`notebooks/eda.ipynb`: record-type check, constant-column staging decision, daily-average seasonal plot with solstices; weather correlation not started)
5. **Feature engineering** — lags, rolling stats, calendar/weather joins, modularized into `src/features/`
6. **Model training & eval** — baseline model first, iterate, all logged in MLflow
7. **Model registry** — promote best model via MLflow Model Registry
8. **API + containerization** — FastAPI wrapper, Dockerfile
9. **(Stretch) SageMaker deployment**

## Resolved Decisions
- **ISO/region:** ISO New England (`ISNE`), pulled via the EIA Open Data API rather than a direct ISO API — chosen for personal relevance (covers Connecticut) over PJM/ERCOT/CAISO.
- **Compute:** workspace only supports serverless (no classic clusters) — job tasks are defined without an explicit cluster/environment spec.

## Open Decisions
- Exact prediction horizon (hourly vs. day-ahead)
- Whether to extend into price (LMP) forecasting as a phase-2 model chained off demand predictions
- Weather data source (Open-Meteo vs. alternatives) — not yet integrated
