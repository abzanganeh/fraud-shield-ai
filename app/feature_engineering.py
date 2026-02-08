"""
Pandas-based feature engineering for inference.

Replicates the Spark pipeline from NB02 in pure pandas/numpy so the
Streamlit app can accept raw transaction CSVs without needing a Spark session.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    """Holds the encoding mappings and scaler parameters extracted from the
    fitted Spark pipeline.  Loaded once at startup."""

    vector_feature_order: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    categorical_mappings: Dict[str, Dict[str, int]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "InferenceConfig":
        with open(path, "rb") as fh:
            raw: Dict[str, Any] = pickle.load(fh)
        return cls(
            vector_feature_order=raw["vector_feature_order"],
            categorical_features=raw["categorical_features"],
            numerical_features=raw["numerical_features"],
            categorical_mappings=raw["categorical_mappings"],
            scaler_mean=np.array(raw["scaler_mean"], dtype=np.float64),
            scaler_scale=np.array(raw["scaler_scale"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Constants (match NB02 definitions)
# ---------------------------------------------------------------------------
HIGH_RISK_CATEGORIES = frozenset(
    ["grocery_pos", "gas_transport", "shopping_pos", "misc_pos", "grocery_net"]
)
HIGH_AMOUNT_BINS = frozenset([">$1000", "$500-$1000", "$300-$500"])
ONLINE_CATEGORIES = frozenset(["shopping_net", "misc_net"])

REQUIRED_RAW_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
]


# ---------------------------------------------------------------------------
# Haversine (vectorised numpy)
# ---------------------------------------------------------------------------
def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Row-level features
# ---------------------------------------------------------------------------
def add_row_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features that only depend on the current row (no aggregation)."""
    df = df.copy()

    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek + 1  # 1=Mon .. 7=Sun (matches Spark dayofweek)
    df["month"] = ts.dt.month

    df["time_bin"] = np.select(
        [
            df["hour"].between(6, 11),
            df["hour"].between(12, 17),
            df["hour"].between(18, 23),
        ],
        ["Morning", "Afternoon", "Evening"],
        default="Night",
    )

    df["is_peak_fraud_hour"] = df["hour"].between(18, 23).astype(int)
    df["is_peak_fraud_day"] = df["day_of_week"].isin([4, 5, 6]).astype(int)
    df["is_peak_fraud_season"] = df["month"].isin([1, 2]).astype(int)
    df["is_high_risk_category"] = df["category"].isin(HIGH_RISK_CATEGORIES).astype(int)

    # Amount bin and city size (intermediate, used for interaction features)
    df["amount_bin"] = np.select(
        [
            df["amt"] > 1000,
            df["amt"] > 500,
            df["amt"] > 300,
            df["amt"] > 100,
            df["amt"] > 50,
        ],
        [">$1000", "$500-$1000", "$300-$500", "$100-$300", "$50-$100"],
        default="<$50",
    )
    df["city_size"] = np.select(
        [
            df["city_pop"] > 1_000_000,
            df["city_pop"] > 500_000,
            df["city_pop"] > 100_000,
            df["city_pop"] > 10_000,
        ],
        ["Metropolitan", "Large City", "Medium City", "Small City"],
        default="Small Town",
    )

    df["customer_merchant_distance_km"] = _haversine_km(
        df["lat"].values,
        df["long"].values,
        df["merch_lat"].values,
        df["merch_long"].values,
    )

    return df


# ---------------------------------------------------------------------------
# Card-level & velocity features (point-in-time windows in pandas)
# ---------------------------------------------------------------------------
def add_card_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute card-level aggregation features using backward-only windows.

    The dataframe must be sorted by unix_time before calling this.
    """
    df = df.sort_values("unix_time").copy()

    # Cumulative count per card (point-in-time)
    df["transaction_count"] = df.groupby("cc_num").cumcount() + 1

    # Card age in days (from first known txn for this card, backward-only)
    ts = pd.to_datetime(df["trans_date_trans_time"])
    # Use unix_time (numeric) for cumulative min, then convert back
    first_unix = df.groupby("cc_num")["unix_time"].cummin()
    df["card_age_days"] = ((df["unix_time"] - first_unix) / 86400).astype(int).clip(lower=0)

    # Velocity: txn count in last 1h / 24h
    df["txn_count_last_1h"] = _rolling_count_by_time(df, "cc_num", "unix_time", 3600)
    df["txn_count_last_24h"] = _rolling_count_by_time(df, "cc_num", "unix_time", 86400)

    # Amount relative to card's historical average (excluding current row)
    df["amt_relative_to_avg"] = df.groupby("cc_num")["amt"].transform(
        lambda s: s / s.expanding().mean().shift(1).fillna(1.0).replace(0, 1.0)
    )

    # --- NEW SEQUENCE FEATURES ---

    # Time since last transaction (seconds)
    df["time_since_last_txn_seconds"] = (
        df.groupby("cc_num")["unix_time"].diff().fillna(0).clip(lower=0).astype(float)
    )

    # Amount deviation from expanding median (more robust than mean)
    df["amt_deviation_from_median"] = df.groupby("cc_num")["amt"].transform(
        lambda s: s / s.expanding().median().shift(1).fillna(1.0).replace(0, 1.0)
    )

    # Rolling std of amount over last 10 transactions
    df["amt_rolling_std"] = df.groupby("cc_num")["amt"].transform(
        lambda s: s.rolling(10, min_periods=1).std().fillna(0.0)
    )

    # Is new merchant (first time this card uses this merchant)
    df["is_new_merchant"] = (df.groupby(["cc_num", "merchant"]).cumcount() == 0).astype(int)

    # Distinct categories in last 24h
    df["category_nunique_last_24h"] = _rolling_nunique_by_time(
        df, "cc_num", "unix_time", "category", 86400
    )

    # Geographic velocity (km/h) -- impossible-travel detector
    df["geo_velocity_kmh"] = np.where(
        df["time_since_last_txn_seconds"] > 0,
        df["customer_merchant_distance_km"] / (df["time_since_last_txn_seconds"] / 3600.0),
        0.0,
    )

    # Amount relative to personal max (expanding, excluding current)
    df["amt_to_personal_max"] = df.groupby("cc_num")["amt"].transform(
        lambda s: s / s.expanding().max().shift(1).fillna(1.0).replace(0, 1.0)
    )

    return df


def _rolling_count_by_time(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    window_seconds: int,
) -> pd.Series:
    """Count rows within a backward time window per group.

    Uses numpy searchsorted for O(n log n) per group instead of O(n^2).
    """
    result = np.ones(len(df), dtype=np.int64)

    for _, grp in df.groupby(group_col):
        times = grp[time_col].values.astype(np.int64)
        lower_bounds = times - window_seconds
        # searchsorted gives the index of the first element >= lower_bound
        left_indices = np.searchsorted(times, lower_bounds, side="left")
        counts = np.arange(1, len(times) + 1) - left_indices
        result[grp.index.values] = counts

    return pd.Series(result, index=df.index)


def _rolling_nunique_by_time(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_col: str,
    window_seconds: int,
) -> pd.Series:
    """Count unique values of *value_col* within a backward time window per group."""
    result = np.ones(len(df), dtype=np.int64)

    for _, grp in df.groupby(group_col):
        times = grp[time_col].values.astype(np.int64)
        values = grp[value_col].values
        for i in range(len(times)):
            lower_bound = times[i] - window_seconds
            left_idx = int(np.searchsorted(times, lower_bound, side="left"))
            result[grp.index.values[i]] = len(set(values[left_idx : i + 1]))

    return pd.Series(result, index=df.index)


# ---------------------------------------------------------------------------
# Interaction & risk features
# ---------------------------------------------------------------------------
def add_interaction_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interaction features and composite risk scores."""
    df = df.copy()

    # Bins from card-level features
    df["card_age_bin"] = np.select(
        [
            df["card_age_days"] < 7,
            df["card_age_days"] < 30,
            df["card_age_days"] < 90,
            df["card_age_days"] < 180,
        ],
        ["<7 days", "7-30 days", "30-90 days", "90-180 days"],
        default="180+ days",
    )
    df["transaction_count_bin"] = np.select(
        [
            df["transaction_count"] <= 5,
            df["transaction_count"].between(6, 20),
            df["transaction_count"].between(21, 100),
        ],
        ["1-5", "6-20", "21-100"],
        default="100+",
    )
    df["is_new_card"] = (df["card_age_days"] <= 30).astype(int)
    df["is_low_volume_card"] = (df["transaction_count"] <= 5).astype(int)

    # Interaction features
    evening = df["time_bin"] == "Evening"
    high_amt = df["amount_bin"].isin(HIGH_AMOUNT_BINS)
    online = df["category"].isin(ONLINE_CATEGORIES)

    df["evening_high_amount"] = (evening & high_amt).astype(int)
    df["evening_online_shopping"] = (evening & online).astype(int)
    df["large_city_evening"] = ((df["city_size"] == "Large City") & evening).astype(int)
    df["new_card_evening"] = ((df["is_new_card"] == 1) & evening).astype(int)
    df["high_amount_online"] = (high_amt & online).astype(int)

    # Risk scores
    df["temporal_risk_score"] = (
        df["is_peak_fraud_hour"] * 0.4
        + df["is_peak_fraud_day"] * 0.3
        + df["is_peak_fraud_season"] * 0.3
    )
    df["geographic_risk_score"] = np.select(
        [
            df["city_pop"] < 10_000,
            df["city_pop"] < 50_000,
            df["city_pop"] < 100_000,
        ],
        [0.3, 0.2, 0.1],
        default=0.0,
    )
    df["card_risk_score"] = (
        df["is_new_card"] * 0.5
        + df["is_low_volume_card"] * 0.3
        + (df["card_age_days"] < 7).astype(float) * 0.2
    )
    total_risk = (
        df["temporal_risk_score"]
        + df["geographic_risk_score"]
        + df["card_risk_score"]
    )
    df["risk_tier"] = np.select(
        [total_risk >= 0.8, total_risk >= 0.4],
        ["High", "Medium"],
        default="Low",
    )

    return df


# ---------------------------------------------------------------------------
# Encode & scale
# ---------------------------------------------------------------------------
def encode_and_scale(
    df: pd.DataFrame,
    config: InferenceConfig,
) -> np.ndarray:
    """Convert an engineered DataFrame into the scaled numeric matrix that
    the XGBoost model expects.

    Returns an (n_rows, 30) float64 array in the same feature order as the
    training data.
    """
    n = len(df)
    n_features = len(config.vector_feature_order)
    out = np.zeros((n, n_features), dtype=np.float64)

    for idx, feat in enumerate(config.vector_feature_order):
        if feat in config.categorical_features:
            mapping = config.categorical_mappings[feat]
            # Unknown categories get the highest index + 1 (Spark handleInvalid="keep")
            fallback = max(mapping.values()) + 1
            out[:, idx] = df[feat].map(mapping).fillna(fallback).values
        else:
            out[:, idx] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0).values

    # Standard scaling
    out = (out - config.scaler_mean) / config.scaler_scale

    return out


# ---------------------------------------------------------------------------
# Full pipeline: raw CSV -> model-ready matrix
# ---------------------------------------------------------------------------
def preprocess_raw(
    raw_df: pd.DataFrame,
    config: InferenceConfig,
) -> np.ndarray:
    """End-to-end: raw transaction DataFrame -> scaled feature matrix."""
    df = add_row_level_features(raw_df)
    df = add_card_velocity_features(df)
    df = add_interaction_risk_features(df)
    return encode_and_scale(df, config)


def validate_raw_columns(df: pd.DataFrame) -> List[str]:
    """Return list of missing required columns."""
    return [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
