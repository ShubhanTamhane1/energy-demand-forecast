# Databricks notebook source
# MAGIC %md
# MAGIC # ISO-NE Demand Forecast — Feature Selection + Regularization + HPO
# MAGIC
# MAGIC Three experiments in one run, all registered as new versions of
# MAGIC `workspace.default.iso_ne_demand_forecast`:
# MAGIC
# MAGIC 1. Linear regression restricted to the features found statistically
# MAGIC    significant (p < 0.05) in the OLS fit -- drops `precipitation`, `rain`,
# MAGIC    `snowfall`, `did_snow`.
# MAGIC 2. ElasticNet with Optuna-tuned `alpha`/`l1_ratio` -- generalizes Ridge
# MAGIC    (l1_ratio=0) and Lasso (l1_ratio=1) in one search.
# MAGIC 3. Optuna-tuned Random Forest, XGBoost, and SVM (all untuned/default last
# MAGIC    time -- SVM in particular did badly on defaults, R2=-0.26).
# MAGIC
# MAGIC **Split strategy:** the outer 80/20 chronological split gives the final
# MAGIC holdout `test_df`, matching prior runs so results are comparable. The
# MAGIC training portion is further split 80/20 chronologically into
# MAGIC `tune_train`/`tune_val`, used only inside Optuna objectives -- `test_df` is
# MAGIC never touched until final evaluation, so hyperparameter selection can't leak
# MAGIC information from the holdout set.

# COMMAND ----------

import json
import time

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/shubhant99@gmail.com/energy-demand-forecast/mlflow_experiment")

CATALOG = "workspace"
SCHEMA = "default"
MODEL_NAME = "iso_ne_demand_forecast"
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

TARGET = "demand_mwh"
DROP_COLS = ["period", TARGET, "_rescued_data"]

# Statistically significant (p < 0.05) in the full OLS fit from the prior
# analysis -- excludes precipitation, rain, snowfall, did_snow (all p > 0.05).
SIGNIFICANT_FEATURES = [
    "temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "did_rain",
    "heating_degree_hours", "cooling_degree_hours",
    "lag_24h", "lag_48h", "lag_168h", "rolling_24h_mean", "rolling_168h_mean",
]

N_TRIALS = 20

# For reference -- the best result from the prior (untuned) comparison run.
PRIOR_BEST = {
    "name": "linear_regression (untuned, all features)",
    "version": "6",
    "mape": 0.0800,
    "rmse": 1440.52,
    "r2": 0.7613,
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load + clean + split

# COMMAND ----------

df = spark.table(f"{CATALOG}.{SCHEMA}.iso_ne_features").toPandas()
df = df.sort_values("period").reset_index(drop=True)

before = len(df)
df = df.dropna(subset=["lag_168h", "rolling_168h_mean"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows lacking a full week of lag history (expected 168).")

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

tune_split_idx = int(len(train_df) * 0.8)
tune_train_df = train_df.iloc[:tune_split_idx]
tune_val_df = train_df.iloc[tune_split_idx:]

all_feature_cols = [c for c in df.columns if c not in DROP_COLS]


def xy(frame, cols):
    return frame[cols], frame[TARGET]


X_train, y_train = xy(train_df, all_feature_cols)
X_test, y_test = xy(test_df, all_feature_cols)
X_tune_train, y_tune_train = xy(tune_train_df, all_feature_cols)
X_tune_val, y_tune_val = xy(tune_val_df, all_feature_cols)
X_train_sig, _ = xy(train_df, SIGNIFICANT_FEATURES)
X_test_sig, _ = xy(test_df, SIGNIFICANT_FEATURES)

print(
    f"Train: {len(train_df)}  Test: {len(test_df)}  "
    f"(tune_train: {len(tune_train_df)}  tune_val: {len(tune_val_df)})"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------


def log_and_register(name, model, X_tr, X_te, y_te, fit_time, params=None):
    predict_start = time.perf_counter()
    preds = model.predict(X_te)
    predict_time = time.perf_counter() - predict_start

    metrics = {
        "rmse": mean_squared_error(y_te, preds) ** 0.5,
        "mae": mean_absolute_error(y_te, preds),
        "mape": mean_absolute_percentage_error(y_te, preds),
        "r2": r2_score(y_te, preds),
    }
    print(
        f"{name}: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
        f"MAPE={metrics['mape']:.4f}  R2={metrics['r2']:.4f}  fit={fit_time:.2f}s"
    )

    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metric("test_rmse", metrics["rmse"])
        mlflow.log_metric("test_mae", metrics["mae"])
        mlflow.log_metric("test_mape", metrics["mape"])
        mlflow.log_metric("test_r2", metrics["r2"])
        mlflow.log_metric("fit_time_sec", fit_time)
        mlflow.log_metric("predict_time_sec", predict_time)

        sample = X_tr.iloc[:5]
        signature = infer_signature(sample, model.predict(sample))
        log_fn = mlflow.xgboost.log_model if isinstance(model, XGBRegressor) else mlflow.sklearn.log_model
        info = log_fn(
            model,
            name="model",
            registered_model_name=FULL_MODEL_NAME,
            signature=signature,
            input_example=sample,
        )

    return {
        "name": name,
        "version": info.registered_model_version,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        **metrics,
    }


results = []

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Linear regression — statistically significant features only

# COMMAND ----------

fit_start = time.perf_counter()
lr_sig = LinearRegression().fit(X_train_sig, y_train)
fit_time = time.perf_counter() - fit_start
results.append(
    log_and_register(
        "linear_regression_significant_features",
        lr_sig,
        X_train_sig,
        X_test_sig,
        y_test,
        fit_time,
        params={"n_features": len(SIGNIFICANT_FEATURES)},
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Regularized linear regression (ElasticNet) — tune alpha + l1_ratio

# COMMAND ----------


def elasticnet_objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 100, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    model = Pipeline(
        [("scaler", StandardScaler()), ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42))]
    )
    with mlflow.start_run(run_name="elasticnet_trial", nested=True):
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        model.fit(X_tune_train, y_tune_train)
        val_mape = mean_absolute_percentage_error(y_tune_val, model.predict(X_tune_val))
        mlflow.log_metric("val_mape", val_mape)
    return val_mape


with mlflow.start_run(run_name="elasticnet_hpo"):
    study_enet = optuna.create_study(direction="minimize")
    study_enet.optimize(elasticnet_objective, n_trials=N_TRIALS)

print(f"Best ElasticNet params: {study_enet.best_params}, val MAPE={study_enet.best_value:.4f}")
best_enet = Pipeline(
    [("scaler", StandardScaler()), ("enet", ElasticNet(**study_enet.best_params, max_iter=10000, random_state=42))]
)
fit_start = time.perf_counter()
best_enet.fit(X_train, y_train)
fit_time = time.perf_counter() - fit_start
results.append(
    log_and_register("linear_regression_elasticnet_tuned", best_enet, X_train, X_test, y_test, fit_time, params=study_enet.best_params)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Random Forest — Optuna tuned

# COMMAND ----------


def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    with mlflow.start_run(run_name="random_forest_trial", nested=True):
        mlflow.log_params(params)
        model.fit(X_tune_train, y_tune_train)
        val_mape = mean_absolute_percentage_error(y_tune_val, model.predict(X_tune_val))
        mlflow.log_metric("val_mape", val_mape)
    return val_mape


with mlflow.start_run(run_name="random_forest_hpo"):
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(rf_objective, n_trials=N_TRIALS)

print(f"Best RF params: {study_rf.best_params}, val MAPE={study_rf.best_value:.4f}")
best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
fit_start = time.perf_counter()
best_rf.fit(X_train, y_train)
fit_time = time.perf_counter() - fit_start
results.append(log_and_register("random_forest_tuned", best_rf, X_train, X_test, y_test, fit_time, params=study_rf.best_params))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. XGBoost — Optuna tuned

# COMMAND ----------


def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = XGBRegressor(**params, random_state=42)
    with mlflow.start_run(run_name="xgboost_trial", nested=True):
        mlflow.log_params(params)
        model.fit(X_tune_train, y_tune_train)
        val_mape = mean_absolute_percentage_error(y_tune_val, model.predict(X_tune_val))
        mlflow.log_metric("val_mape", val_mape)
    return val_mape


with mlflow.start_run(run_name="xgboost_hpo"):
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS)

print(f"Best XGBoost params: {study_xgb.best_params}, val MAPE={study_xgb.best_value:.4f}")
best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42)
fit_start = time.perf_counter()
best_xgb.fit(X_train, y_train)
fit_time = time.perf_counter() - fit_start
results.append(log_and_register("xgboost_tuned", best_xgb, X_train, X_test, y_test, fit_time, params=study_xgb.best_params))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. SVM (SVR) — Optuna tuned

# COMMAND ----------


def svm_objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
    }
    model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", **params))])
    with mlflow.start_run(run_name="svm_trial", nested=True):
        mlflow.log_params(params)
        model.fit(X_tune_train, y_tune_train)
        val_mape = mean_absolute_percentage_error(y_tune_val, model.predict(X_tune_val))
        mlflow.log_metric("val_mape", val_mape)
    return val_mape


with mlflow.start_run(run_name="svm_hpo"):
    study_svm = optuna.create_study(direction="minimize")
    study_svm.optimize(svm_objective, n_trials=N_TRIALS)

print(f"Best SVM params: {study_svm.best_params}, val MAPE={study_svm.best_value:.4f}")
best_svm = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", **study_svm.best_params))])
fit_start = time.perf_counter()
best_svm.fit(X_train, y_train)
fit_time = time.perf_counter() - fit_start
results.append(log_and_register("svm_tuned", best_svm, X_train, X_test, y_test, fit_time, params=study_svm.best_params))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare against prior best and promote

# COMMAND ----------

print(f"Prior best (for reference): {PRIOR_BEST}")
for r in results:
    print(
        f"{r['name']}: MAPE={r['mape']:.4f}  RMSE={r['rmse']:.2f}  R2={r['r2']:.4f}  "
        f"fit={r['fit_time_sec']:.2f}s"
    )

best = min(results, key=lambda r: r["mape"])
print(f"\nBest of this run: {best['name']} (version {best['version']}), MAPE={best['mape']:.4f}")

client = MlflowClient(registry_uri="databricks-uc")
if best["mape"] < PRIOR_BEST["mape"]:
    client.set_registered_model_alias(FULL_MODEL_NAME, "prod", best["version"])
    print(f"New best beats prior best -- {FULL_MODEL_NAME} version {best['version']} aliased @prod")
else:
    print(f"Prior best (MAPE={PRIOR_BEST['mape']:.4f}) still wins -- @prod alias unchanged")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"results": results, "best": best, "prior_best": PRIOR_BEST}))
