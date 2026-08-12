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
- **ISO/grid operator API** (PJM / ERCOT / CAISO — TBD, pending API access check) for historical + near-real-time load data.
- **Weather data** — temperature is the dominant driver of load (heating/cooling degree days). Source TBD (e.g., Open-Meteo, no API key required).
- **Calendar features** — day-of-week, hour-of-day, holidays — engineered, not pulled from an external source.

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
- **Platform:** Databricks (Repos for git-backed notebooks, Delta Lake for storage, scheduled Jobs for ingestion/pipeline orchestration)
- **Experiment tracking:** MLflow (Databricks-managed — no separate server needed)
- **Modeling:** traditional ML (e.g., gradient boosting / regression models) via scikit-learn or similar — no deep learning needed for this problem
- **API:** FastAPI for model serving
- **Containerization:** Docker
- **Deployment (stretch goal):** AWS SageMaker endpoint
- **Version control:** Git, connected to Databricks Repos

## Repo Structure
```
energy-demand-forecast/
├── data/                  # gitignored — raw/interim/processed
├── notebooks/             # exploration only, not production logic
├── src/
│   ├── ingestion/         # ISO API polling scripts
│   ├── features/          # cleaning + feature engineering functions
│   ├── models/            # training/eval scripts
│   └── serving/           # FastAPI app for model serving
├── tests/
├── mlflow/                # experiment configs if needed
├── requirements.txt
├── README.md
└── .gitignore
```

## Development Phases
1. **Setup** — repo skeleton, README, connect Databricks Repos to GitHub
2. **Data access spike** — validate ISO API + weather API access before building anything else
3. **Ingestion pipeline** — historical backfill + scheduled polling job into Delta
4. **EDA** — explore load patterns, seasonality, weather correlation
5. **Feature engineering** — lags, rolling stats, calendar/weather joins, modularized into `src/features/`
6. **Model training & eval** — baseline model first, iterate, all logged in MLflow
7. **Model registry** — promote best model via MLflow Model Registry
8. **API + containerization** — FastAPI wrapper, Dockerfile
9. **(Stretch) SageMaker deployment**

## Open Decisions
- Which ISO to pull from (PJM vs. ERCOT vs. CAISO) — depends on API access ease, needs a quick validation check
- Exact prediction horizon (hourly vs. day-ahead)
- Whether to extend into price (LMP) forecasting as a phase-2 model chained off demand predictions
