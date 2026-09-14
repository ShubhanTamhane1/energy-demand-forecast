# Databricks notebook source
# Hourly ingestion job: pulls a rolling recent window of Boston hourly weather
# (proxy point for ISO-NE) from the Open-Meteo Forecast API and merges it into
# a Delta table.
#
# Uses the Forecast API, not the Historical Weather API used for the
# `data/raw/raw_weather_data.ipynb` backfill: the Historical (ERA5 reanalysis)
# API has ~5 days of latency and can't serve "just happened" data. The
# Forecast API's `past_days` param returns recent actuals instead.
#
# `forecast_days=1` (the minimum) still returns the full current day
# regardless of what time the job runs, so any hour later than "now" is
# explicitly filtered out below -- this table should only ever contain
# weather that has already happened, never a forecast.
#
# No API key needed -- Open-Meteo's endpoints used here are unauthenticated.

import json
import urllib.parse
import urllib.request

import pandas as pd

BASE_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE = 42.36  # Boston -- proxy point for ISO-NE
LONGITUDE = -71.06
TIMEZONE = "America/New_York"
PAST_DAYS = 2  # matches the demand job's 48h lookback
FORECAST_DAYS = 1  # minimum allowed; see note above on filtering

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "temperature_2m,precipitation,rain,snowfall",
    "timezone": TIMEZONE,
    "past_days": PAST_DAYS,
    "forecast_days": FORECAST_DAYS,
}
url = BASE_URL + "?" + urllib.parse.urlencode(params)
with urllib.request.urlopen(url, timeout=30) as resp:
    data = json.loads(resp.read())

hourly = data["hourly"]
print(f"Pulled {len(hourly['time'])} hourly records for {LATITUDE}N {LONGITUDE}E")

pdf = pd.DataFrame(
    {
        "period": hourly["time"],
        "temperature_2m": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "rain": hourly["rain"],
        "snowfall": hourly["snowfall"],
    }
)

# Times come back as naive local (TIMEZONE) strings -- localize, then convert
# to UTC to match the demand table's period column.
pdf["period"] = (
    pd.to_datetime(pdf["period"]).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")
)

# Drop any hour later than now: forecast_days=1 returns the full current day
# regardless of run time, so the tail can still be forecast, not actuals.
now = pd.Timestamp.now(tz="UTC")
pdf = pdf[pdf["period"] <= now].reset_index(drop=True)
pdf["period"] = pdf["period"].dt.tz_localize(None)

print(f"Rows after dropping future hours: {len(pdf)}")

# COMMAND ----------

df = spark.createDataFrame(pdf)

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "default"
TABLE = f"{CATALOG}.{SCHEMA}.boston_weather"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    period TIMESTAMP,
    temperature_2m DOUBLE,
    precipitation DOUBLE,
    rain DOUBLE,
    snowfall DOUBLE
)
USING DELTA
""")

df.createOrReplaceTempView("incoming")

spark.sql(f"""
MERGE INTO {TABLE} AS target
USING incoming AS source
ON target.period = source.period
WHEN MATCHED THEN UPDATE SET
    target.temperature_2m = source.temperature_2m,
    target.precipitation = source.precipitation,
    target.rain = source.rain,
    target.snowfall = source.snowfall
WHEN NOT MATCHED THEN INSERT (period, temperature_2m, precipitation, rain, snowfall)
    VALUES (source.period, source.temperature_2m, source.precipitation, source.rain, source.snowfall)
""")

print(f"Merged into {TABLE}")
