"""Downloads the @prod model artifact from the Unity Catalog registry so it
can be baked into the serving Docker image.

Run before `docker build` (from the repo root):

    DATABRICKS_HOST=https://dbc-fabd11bb-a1ee.cloud.databricks.com \\
    DATABRICKS_TOKEN=<token> \\
    python src/serving/download_model.py

Requires DATABRICKS_HOST/DATABRICKS_TOKEN in the environment -- these are
read by mlflow's `databricks` tracking URI to authenticate.
"""

import shutil
from pathlib import Path

import mlflow

MODEL_URI = "models:/workspace.default.iso_ne_demand_forecast@prod"
DEST = Path(__file__).parent / "model"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

if DEST.exists():
    shutil.rmtree(DEST)

local_path = mlflow.artifacts.download_artifacts(MODEL_URI, dst_path=str(DEST))
print(f"Downloaded {MODEL_URI} to {local_path}")
