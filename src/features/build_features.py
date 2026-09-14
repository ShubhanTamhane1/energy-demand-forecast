"""Calendar and weather feature engineering.

Takes the joined demand+weather table (see
data/interim/join_demand_weather.ipynb) and adds derived features. Input
`period` is expected UTC (naive or tz-aware) -- calendar features are derived
in America/New_York local time since demand patterns (commute hours,
weekday/weekend) are a local-time phenomenon, not a UTC one.
"""

import holidays
import numpy as np
import pandas as pd

LOCAL_TZ = "America/New_York"
HDD_CDD_BASE_F = 65.0

us_holidays = holidays.US()


def add_calendar_features(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    df = df.copy()
    period_utc = df[period_col]
    if period_utc.dt.tz is None:
        period_utc = period_utc.dt.tz_localize("UTC")
    local = period_utc.dt.tz_convert(LOCAL_TZ)

    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek  # Monday=0 .. Sunday=6
    df["month"] = local.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = local.dt.date.astype("O").apply(lambda d: d in us_holidays)

    # Cyclical encodings so the model sees hour 23 and hour 0 (or Sun and Mon)
    # as adjacent rather than maximally distant.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["did_rain"] = df["rain"] > 0
    df["did_snow"] = df["snowfall"] > 0

    temperature_f = df["temperature_2m"] * 9 / 5 + 32
    df["heating_degree_hours"] = (HDD_CDD_BASE_F - temperature_f).clip(lower=0)
    df["cooling_degree_hours"] = (temperature_f - HDD_CDD_BASE_F).clip(lower=0)

    return df


def add_demand_features(
    df: pd.DataFrame, demand_col: str = "demand_mwh", period_col: str = "period"
) -> pd.DataFrame:
    """Lag and rolling-mean features derived from demand itself.

    Requires df sorted ascending by `period_col`. Rolling means are computed
    on `shift(1)` (i.e. exclude the current hour) so they only ever look at
    demand that would already be known at prediction time -- same reasoning
    as the lags. 24h and 168h windows match the daily/weekly lag periods
    rather than introducing a third, unrelated window.
    """
    df = df.sort_values(period_col).reset_index(drop=True)

    df["lag_24h"] = df[demand_col].shift(24)
    df["lag_48h"] = df[demand_col].shift(48)
    df["lag_168h"] = df[demand_col].shift(168)

    trailing = df[demand_col].shift(1)
    df["rolling_24h_mean"] = trailing.rolling(24).mean()
    df["rolling_168h_mean"] = trailing.rolling(168).mean()

    return df


def build_features(
    df: pd.DataFrame, period_col: str = "period", demand_col: str = "demand_mwh"
) -> pd.DataFrame:
    df = add_calendar_features(df, period_col=period_col)
    df = add_weather_features(df)
    df = add_demand_features(df, demand_col=demand_col, period_col=period_col)
    return df
