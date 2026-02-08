"""
Fraud Shield AI -- Streamlit Web Application

Upload a raw transaction CSV (same schema as fraudTrain.csv / fraudTest.csv),
and the app will run the full feature-engineering + prediction pipeline
in real time.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import (
    InferenceConfig,
    preprocess_raw,
    validate_raw_columns,
    REQUIRED_RAW_COLUMNS,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "input"
SAMPLE_CSV_PATH = APP_DIR / "sample_transactions.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Shield AI",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> Tuple:
    """Load XGBoost model and metadata."""
    model_path = MODELS_DIR / "xgb_best_model.pkl"
    metadata_path = MODELS_DIR / "xgb_best_model_metadata.pkl"

    if not model_path.exists():
        st.error(
            "Model file not found. Run notebook 04 first to train and save the model."
        )
        st.stop()

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
    return model, metadata


@st.cache_resource
def load_two_stage_config() -> Optional[Dict]:
    """Load two-stage fraud detection config if available."""
    config_path = MODELS_DIR / "two_stage_config.pkl"
    if config_path.exists():
        import pickle
        with open(config_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_inference_config() -> InferenceConfig:
    """Load the inference configuration (scaler params, encodings)."""
    config_path = MODELS_DIR / "inference_config.pkl"
    if not config_path.exists():
        st.error(
            "Inference config not found. Ensure models/inference_config.pkl exists."
        )
        st.stop()
    return InferenceConfig.load(config_path)


@st.cache_data
def load_comparison() -> Optional[pd.DataFrame]:
    """Load model comparison CSV if available."""
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def get_sample_csv_bytes() -> Optional[bytes]:
    """Return a small sample CSV for users to download as a template."""
    if SAMPLE_CSV_PATH.exists():
        return SAMPLE_CSV_PATH.read_bytes()
    # Fallback: try to extract from fraudTest.csv
    for candidate in [DATA_DIR / "fraudTest.csv", DATA_DIR / "fraudTrain.csv"]:
        if candidate.exists():
            sample = pd.read_csv(candidate, nrows=20)
            return sample.to_csv(index=False).encode("utf-8")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict(
    model,
    X: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, binary_flags)."""
    probas = model.predict_proba(X)[:, 1]
    flags = (probas >= threshold).astype(int)
    return probas, flags


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    model, metadata = load_model()
    config = load_inference_config()
    comparison_df = load_comparison()
    two_stage = load_two_stage_config()

    threshold = metadata.get("optimal_threshold", 0.5)

    # Two-stage defaults
    stage1_thresh = two_stage["stage1_threshold"] if two_stage else threshold
    auto_block_thresh = two_stage["stage2_auto_block_threshold"] if two_stage else 0.90

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("Fraud Shield AI")
        st.markdown("---")
        st.subheader("Model Info")
        st.write(f"**Model:** {metadata.get('model_name', 'XGBoost')}")
        st.write(f"**Features:** {metadata.get('n_features', len(config.vector_feature_order))}")

        if two_stage:
            st.markdown("---")
            st.subheader("Two-Stage Thresholds")
            stage1_thresh = st.slider(
                "Stage 1 (flag threshold)",
                min_value=0.0, max_value=1.0,
                value=float(stage1_thresh), step=0.01,
                help="Transactions >= this threshold are flagged for review or blocking.",
            )
            auto_block_thresh = st.slider(
                "Stage 2 (auto-block threshold)",
                min_value=0.0, max_value=1.0,
                value=float(auto_block_thresh), step=0.01,
                help="Flagged transactions >= this threshold are auto-blocked.",
            )
        else:
            st.write(f"**Default threshold:** {threshold:.4f}")
            st.markdown("---")
            threshold = st.slider(
                "Decision threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(threshold),
                step=0.01,
                help="Transactions with fraud probability >= threshold will be flagged.",
            )

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------
    st.title("Fraud Shield AI -- Transaction Fraud Detector")
    st.markdown(
        "Upload a **raw transaction CSV** (same format as `fraudTrain.csv` / "
        "`fraudTest.csv`). The app handles all feature engineering and "
        "preprocessing automatically."
    )

    tab_predict, tab_format, tab_dashboard, tab_models = st.tabs(
        ["Predict", "Expected CSV Format", "Dashboard", "Model Comparison"]
    )

    # ---- TAB: Expected Format ----
    with tab_format:
        st.subheader("Required CSV Columns")
        st.markdown(
            "Your CSV must contain at least the following columns. "
            "Extra columns are ignored."
        )
        col_info = {
            "trans_date_trans_time": "Transaction timestamp, e.g. `2020-06-21 12:14:25`",
            "cc_num": "Credit card number (used for card-level aggregation)",
            "merchant": "Merchant name (used for is_new_merchant feature)",
            "category": "Merchant category, e.g. `gas_transport`, `grocery_pos`, `shopping_net`",
            "amt": "Transaction amount in dollars",
            "lat": "Customer latitude",
            "long": "Customer longitude",
            "city_pop": "City population",
            "unix_time": "Unix epoch seconds of the transaction",
            "merch_lat": "Merchant latitude",
            "merch_long": "Merchant longitude",
        }
        st.table(pd.DataFrame(
            {"Column": col_info.keys(), "Description": col_info.values()}
        ))

        st.markdown("**Valid `category` values:**")
        valid_cats = sorted(config.categorical_mappings.get("category", {}).keys())
        st.code(", ".join(valid_cats))

        st.markdown("---")
        sample_bytes = get_sample_csv_bytes()
        if sample_bytes:
            st.download_button(
                label="Download sample CSV (20 rows)",
                data=sample_bytes,
                file_name="sample_transactions.csv",
                mime="text/csv",
            )

    # ---- TAB: Predict ----
    with tab_predict:
        uploaded = st.file_uploader(
            "Upload raw transaction CSV",
            type=["csv"],
            help="Same format as fraudTrain.csv or fraudTest.csv",
        )

        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            st.write(
                f"Uploaded **{raw_df.shape[0]:,}** rows, "
                f"**{raw_df.shape[1]}** columns."
            )

            # Validate required columns
            missing = validate_raw_columns(raw_df)
            if missing:
                st.error(
                    f"Missing required columns: **{', '.join(missing)}**. "
                    "Check the 'Expected CSV Format' tab for details."
                )
                st.stop()

            # Feature engineering + scaling
            with st.spinner("Running feature engineering and preprocessing..."):
                X = preprocess_raw(raw_df, config)

            st.success(
                f"Preprocessed {X.shape[0]:,} transactions into "
                f"{X.shape[1]} features."
            )

            # Predict
            probas = model.predict_proba(X)[:, 1]
            results = raw_df.copy()
            results["fraud_probability"] = probas

            if two_stage:
                results["risk_tier"] = "Cleared"
                flagged_mask = probas >= stage1_thresh
                results.loc[flagged_mask & (probas < auto_block_thresh), "risk_tier"] = "Review"
                results.loc[probas >= auto_block_thresh, "risk_tier"] = "Auto-Block"
                results["is_flagged"] = flagged_mask.astype(int)
            else:
                results["is_flagged"] = (probas >= threshold).astype(int)
                results["risk_tier"] = np.where(
                    results["is_flagged"] == 1, "Flagged", "Cleared"
                )

            # Keep only useful display columns
            display_cols = [
                c for c in [
                    "trans_date_trans_time", "cc_num", "merchant", "category",
                    "amt", "city", "state", "fraud_probability", "risk_tier",
                ]
                if c in results.columns
            ]
            if "is_fraud" in results.columns:
                display_cols.append("is_fraud")

            n_total = len(probas)
            n_flagged = int(results["is_flagged"].sum())
            flag_rate = n_flagged / n_total if n_total else 0

            if two_stage:
                n_blocked = int((results["risk_tier"] == "Auto-Block").sum())
                n_review = int((results["risk_tier"] == "Review").sum())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", f"{n_total:,}")
                c2.metric("Auto-Block", f"{n_blocked:,}")
                c3.metric("Review Queue", f"{n_review:,}")
                c4.metric("Cleared", f"{n_total - n_flagged:,}")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", f"{n_total:,}")
                col2.metric("Flagged as Fraud", f"{n_flagged:,}")
                col3.metric("Flag Rate", f"{flag_rate:.2%}")

            st.markdown("---")

            if two_stage:
                st.subheader("Auto-Blocked Transactions")
                blocked = results[results["risk_tier"] == "Auto-Block"].sort_values(
                    "fraud_probability", ascending=False
                )
                if blocked.empty:
                    st.info("No transactions auto-blocked.")
                else:
                    st.dataframe(blocked[display_cols].head(200), use_container_width=True)

                st.subheader("Review Queue")
                review = results[results["risk_tier"] == "Review"].sort_values(
                    "fraud_probability", ascending=False
                )
                if review.empty:
                    st.info("No transactions in review queue.")
                else:
                    st.dataframe(review[display_cols].head(200), use_container_width=True)
            else:
                st.subheader("Flagged Transactions")
                flagged = results[results["is_flagged"] == 1].sort_values(
                    "fraud_probability", ascending=False
                )
                if flagged.empty:
                    st.info("No transactions flagged at the current threshold.")
                else:
                    st.dataframe(flagged[display_cols].head(200), use_container_width=True)

            st.subheader("All Transactions")
            st.dataframe(
                results[display_cols]
                .sort_values("fraud_probability", ascending=False)
                .head(500),
                use_container_width=True,
            )

            # Download full results
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Upload a raw transaction CSV to start. "
                "Check the **Expected CSV Format** tab for column requirements."
            )

    # ---- TAB: Dashboard ----
    with tab_dashboard:
        if uploaded is not None:
            import matplotlib.pyplot as plt

            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(probas, bins=50, edgecolor="black", alpha=0.7)
            if two_stage:
                ax.axvline(stage1_thresh, color="orange", linestyle="--",
                           label=f"Stage 1 ({stage1_thresh:.2f})")
                ax.axvline(auto_block_thresh, color="red", linestyle="--",
                           label=f"Auto-Block ({auto_block_thresh:.2f})")
            else:
                ax.axvline(threshold, color="red", linestyle="--",
                           label=f"Threshold ({threshold:.2f})")
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Risk tier breakdown
            if two_stage:
                st.subheader("Risk Tier Breakdown")
                tier_counts = results["risk_tier"].value_counts()
                fig_tier, ax_tier = plt.subplots(figsize=(6, 4))
                colors = {"Auto-Block": "#d32f2f", "Review": "#ff9800", "Cleared": "#4caf50"}
                bars = ax_tier.bar(
                    tier_counts.index,
                    tier_counts.values,
                    color=[colors.get(t, "#999") for t in tier_counts.index],
                )
                ax_tier.set_ylabel("Count")
                ax_tier.set_title("Transaction Risk Tiers")
                ax_tier.grid(True, alpha=0.3, axis="y")
                for bar, val in zip(bars, tier_counts.values):
                    ax_tier.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                 f"{val:,}", ha="center", va="bottom", fontsize=10)
                st.pyplot(fig_tier)

            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model, "feature_importances_"):
                feat_labels = config.vector_feature_order
                imp = (
                    pd.DataFrame({
                        "feature": feat_labels,
                        "importance": model.feature_importances_,
                    })
                    .sort_values("importance", ascending=True)
                    .tail(15)
                )
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(imp["feature"], imp["importance"])
                ax2.set_xlabel("Importance")
                ax2.set_title("Top 15 Features")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
        else:
            st.info("Upload data in the Predict tab first.")

    # ---- TAB: Model Comparison ----
    with tab_models:
        st.subheader("Supervised Model Comparison")
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No model comparison data found. Run notebook 04 to generate it.")

        val_metrics = metadata.get(
            "val_metrics", metadata.get("metrics", {}).get("val", {})
        )
        test_metrics = metadata.get(
            "test_metrics", metadata.get("metrics", {}).get("test", {})
        )

        if val_metrics:
            st.subheader("Best Model Performance")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Validation**")
                for k, v in val_metrics.items():
                    st.write(f"- {k}: {v:.4f}")
            with c2:
                st.markdown("**Test**")
                for k, v in test_metrics.items():
                    st.write(f"- {k}: {v:.4f}")


if __name__ == "__main__":
    main()
