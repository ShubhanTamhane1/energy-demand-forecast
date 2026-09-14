# ISO-NE Demand Forecast

An end-to-end electricity demand (load) forecasting system for ISO New England — built as a hands-on portfolio project covering the full production ML lifecycle: live hourly ingestion, feature engineering, experiment-tracked model training, and containerized deployment to a real cloud inference endpoint.

Predicts hourly electricity demand (MWh) from historical load, weather, and calendar signals.

**What's predicted:** demand (MWh) for a specific future hour, framed as a **day-ahead (24h-ahead) forecast** — every lag feature (`lag_24h`/`48h`/`168h`) is at least 24 hours old relative to the target hour, so at forecast-issue time all of them are already known; only the target hour's weather and calendar features are needed on top. Calendar features are deterministic (no uncertainty), but weather is not: training and evaluation used the *actual* observed weather for the target hour, not a day-ahead weather forecast, so real-world day-ahead accuracy will also depend on weather forecast quality — something this project hasn't measured yet.

See [`architecture-diagram.html`](./architecture-diagram.html) for the full pipeline diagram, and [`CLAUDE.md`](./CLAUDE.md) for detailed project notes and decision history.

## Architecture

```
EIA API  +  Open-Meteo API
        │  (hourly Databricks job, self-healing MERGE)
        ▼
Delta Lake / Unity Catalog  (iso_ne_demand, boston_weather)
        │  join + feature engineering
        ▼
iso_ne_features  (Delta, UC)
        │  chronological 80/20 split
        ▼
MLflow + Optuna training  →  Unity Catalog Model Registry (@prod)
        │  download artifact
        ▼
FastAPI  →  Docker  →  Amazon ECR  →  SageMaker real-time endpoint
```

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion | EIA Open Data API (ISO-NE hourly demand), Open-Meteo API (Boston hourly weather), Databricks Jobs (hourly, serverless) |
| Storage | Delta Lake, Unity Catalog |
| Feature engineering | pandas, `holidays` (US calendar) |
| Experiment tracking | MLflow (Databricks-managed) |
| Hyperparameter tuning | Optuna |
| Modeling | scikit-learn (Linear Regression, ElasticNet, Random Forest, SVM), XGBoost |
| Model registry | Unity Catalog (movable `@prod` alias) |
| Serving | FastAPI, Docker |
| Deployment | Amazon ECR, AWS SageMaker (real-time endpoint, bring-your-own-container) |
| Infra as code | Databricks Asset Bundles (`databricks.yml`) |

## Data Pipeline

- **Historical backfill:** one year of hourly ISO-NE demand (EIA) and Boston weather (Open-Meteo Historical API), joined and feature-engineered locally, then materialized as a Delta table on Databricks.
- **Live ingestion:** a Databricks job runs hourly with two tasks — one pulls a rolling 48h demand window, the other a rolling weather window (Open-Meteo Forecast API, since the historical/archive API has ~5 days of latency and can't serve recent actuals). Both MERGE (upsert) into their Delta tables, which self-heals against missed runs and catches late-arriving revisions.
- **Features:** calendar (hour/day-of-week/month, weekend/holiday flags, cyclical sin/cos encodings), weather-derived (rain/snow flags, heating/cooling degree hours), and demand lag/rolling features (24h/48h/168h lags, trailing rolling means) — all leakage-safe (rolling means computed on `shift(1)`, train/test split kept strictly chronological).

## Model Results

Five model families, each tuned with Optuna against a chronological inner validation split (never the final test set):

| Model | RMSE (MWh) | MAE (MWh) | MAPE | Fit time |
|---|---|---|---|---|
| **XGBoost (tuned)** — `@prod` | 1,575 | **1,140** | **7.27%** | 0.95s |
| SVM / SVR (tuned) | 1,491 | 1,168 | 7.72% | 1.48s |
| ElasticNet (tuned) | 1,447 | 1,151 | 7.80% | 0.03s |
| Linear Regression (significant features) | 1,436 | 1,166 | 7.98% | 0.07s |
| Random Forest (tuned) | 1,769 | 1,280 | 8.10% | 3.50s |

**XGBoost is `@prod`**, selected on MAPE — the industry-standard metric for load forecasting, since it's scale-free and directly interpretable as % error. Worth noting: Linear Regression/ElasticNet actually edge it out on R². The demand signal turned out to be strongly linear — dominated by `lag_24h` (0.87 correlation with demand on its own) — which is why a well-specified linear model stayed competitive with tuned tree ensembles throughout this project, including beating an *untuned* XGBoost baseline earlier on. SHAP and OLS significance testing (see `CLAUDE.md`) confirm the same two signals drive every model: recent demand (lags/rolling means) and thermal load (heating/cooling degree hours).

## Repo Structure

```
├── data/                  # gitignored — raw/interim/processed backfills
├── notebooks/             # EDA (exploration only)
├── src/
│   ├── ingestion/          # hourly Databricks job tasks (demand, weather)
│   ├── features/           # build_features.py
│   ├── models/              # train_model.py — MLflow + Optuna training
│   └── serving/              # FastAPI app, Dockerfile, model download script
├── architecture-diagram.html
├── databricks.yml          # Databricks Asset Bundle definition
└── requirements.txt
```

## Status

Full lifecycle is built and verified end-to-end, including a live `InvokeEndpoint` call against a real SageMaker endpoint. See `CLAUDE.md`'s **Open Decisions** for what's still unsettled (prediction horizon, whether to extend to price/LMP forecasting, porting feature engineering onto the live Delta tables rather than the historical backfill).
