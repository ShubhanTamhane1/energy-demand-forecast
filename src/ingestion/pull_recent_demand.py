# Databricks notebook source
# Hourly ingestion job: pulls a rolling window of ISO-NE hourly demand from
# the EIA API and merges it into a Delta table. Re-pulling a lookback window
# (not just the latest hour) catches late-arriving/revised EIA readings and
# self-heals if a run is ever missed.
#
# Only `period` and `value` are kept — respondent/type/units are constant
# for this feed and dropped here per the staging decision made during EDA.

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

API_KEY = dbutils.secrets.get(scope="energy-demand-forecast", key="eia_api_key")

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
RESPONDENT = "ISNE"
TYPE = "D"
LOOKBACK_HOURS = 48
PAGE_SIZE = 5000

end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(hours=LOOKBACK_HOURS)


def fetch_page(offset, length=PAGE_SIZE):
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": RESPONDENT,
        "facets[type][]": TYPE,
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


first_page = fetch_page(offset=0)
total_rows = int(first_page["response"]["total"])
all_records = list(first_page["response"]["data"])

offset = PAGE_SIZE
while offset < total_rows:
    page = fetch_page(offset=offset)
    all_records.extend(page["response"]["data"])
    offset += PAGE_SIZE
    time.sleep(0.2)

print(f"Pulled {len(all_records)} records for {start} to {end}")

pdf = pd.DataFrame(all_records)
pdf["period"] = pd.to_datetime(pdf["period"], format="%Y-%m-%dT%H")
pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
pdf = pdf[["period", "value"]]

# COMMAND ----------

df = spark.createDataFrame(pdf)

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "default"
TABLE = f"{CATALOG}.{SCHEMA}.iso_ne_demand"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    period TIMESTAMP,
    value DOUBLE
)
USING DELTA
""")

df.createOrReplaceTempView("incoming")

spark.sql(f"""
MERGE INTO {TABLE} AS target
USING incoming AS source
ON target.period = source.period
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (period, value) VALUES (source.period, source.value)
""")

print(f"Merged into {TABLE}")
