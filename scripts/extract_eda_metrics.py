"""Extract key metrics from EDA result parquet files in data/checkpoints/results/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / "results"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / "eda_metrics_extracted.json"

METRIC_COLS = {"fraud_rate_pct", "total_txns", "fraud_count"}
OPTIONAL_COLS = {"legit_count"}


def _dimension_col(df: pd.DataFrame) -> str | None:
    excl = METRIC_COLS | OPTIONAL_COLS
    cand = [c for c in df.columns if c not in excl]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        for name in ("amount_bin", "hour", "day_name", "month_name", "time_bin", "category",
                     "state", "city_size", "distance_category", "city", "merch_location_key",
                     "age_bin", "gender", "job", "card_age_bin", "transaction_count_bin",
                     "period"):
            if name in cand:
                return name
        return cand[0]
    return None


def _risk_ratio(df: pd.DataFrame) -> float | None:
    if "fraud_rate_pct" not in df.columns or df.empty:
        return None
    rates = df["fraud_rate_pct"].dropna()
    if rates.empty:
        return None
    mx = rates.max()
    mn = rates.min()
    if mn <= 0 or mx <= 0:
        return None
    return round(float(mx / mn), 2)


def extract_metrics(path: Path) -> dict:
    df = pd.read_parquet(path)
    has = METRIC_COLS.intersection(df.columns)
    if METRIC_COLS != has:
        return {"file": path.name, "schema": "non_stats", "columns": list(df.columns)}

    dim = _dimension_col(df)
    total_txns = int(df["total_txns"].sum())
    fraud_count = int(df["fraud_count"].sum())
    overall_rate = round(100.0 * fraud_count / total_txns, 4) if total_txns else None

    peak = df.loc[df["fraud_rate_pct"].idxmax()]
    safest = df.loc[df["fraud_rate_pct"].idxmin()]
    risk_ratio = _risk_ratio(df)

    out: dict = {
        "file": path.name,
        "dimension": dim,
        "total_txns": total_txns,
        "fraud_count": fraud_count,
        "fraud_rate_pct_overall": overall_rate,
        "risk_ratio": risk_ratio,
        "peak": {
            "value": str(peak[dim]) if dim else None,
            "fraud_rate_pct": round(float(peak["fraud_rate_pct"]), 4),
            "total_txns": int(peak["total_txns"]),
            "fraud_count": int(peak["fraud_count"]),
        },
        "safest": {
            "value": str(safest[dim]) if dim else None,
            "fraud_rate_pct": round(float(safest["fraud_rate_pct"]), 4),
            "total_txns": int(safest["total_txns"]),
            "fraud_count": int(safest["fraud_count"]),
        },
        "bins": [],
    }

    for _, row in df.iterrows():
        b: dict = {
            "fraud_rate_pct": round(float(row["fraud_rate_pct"]), 4),
            "total_txns": int(row["total_txns"]),
            "fraud_count": int(row["fraud_count"]),
        }
        if dim:
            b["value"] = str(row[dim])
        out["bins"].append(b)

    return out


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results dir not found: {RESULTS_DIR}")

    paths = sorted(RESULTS_DIR.glob("*.parquet"))
    payload: dict = {"source_dir": str(RESULTS_DIR), "files": []}

    for p in paths:
        try:
            payload["files"].append(extract_metrics(p))
        except Exception as e:
            payload["files"].append({"file": p.name, "error": str(e)})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} ({len(paths)} files processed).")


if __name__ == "__main__":
    main()
