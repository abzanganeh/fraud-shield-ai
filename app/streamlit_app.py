"""
Fraud Shield AI -- Streamlit Web Application

Upload a CSV of transactions, run inference with the best trained model,
and inspect fraud predictions with interactive visualizations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

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
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Tuple:
    """Load the best XGBoost model, its metadata, and feature names."""
    model_path = MODELS_DIR / "xgb_best_model.pkl"
    metadata_path = MODELS_DIR / "xgb_best_model_metadata.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"

    if not model_path.exists():
        st.error(
            "Model file not found. Please run notebook 04 first to train and save the model."
        )
        st.stop()

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    feature_names: Optional[List[str]] = None
    if feature_names_path.exists():
        import pickle

        with open(feature_names_path, "rb") as fh:
            feature_names = pickle.load(fh)

    return model, metadata, feature_names


@st.cache_data
def load_comparison() -> Optional[pd.DataFrame]:
    """Load model comparison CSV if available."""
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def align_features(
    df: pd.DataFrame,
    expected_features: List[str],
) -> pd.DataFrame:
    """Ensure uploaded data has the expected feature columns, filling missing with 0."""
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected_features]


def predict(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Run inference and append prediction columns."""
    X = df[feature_cols].values.astype(np.float32)
    probas = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["fraud_probability"] = probas
    df["is_flagged"] = (probas >= threshold).astype(int)
    return df


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    model, metadata, raw_feature_names = load_model()
    comparison_df = load_comparison()

    threshold = metadata.get("optimal_threshold", 0.5)
    feature_cols = metadata.get("features", raw_feature_names or [])

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("Fraud Shield AI")
        st.markdown("---")
        st.subheader("Model Info")
        st.write(f"**Model:** {metadata.get('model_name', 'XGBoost')}")
        st.write(f"**Features:** {metadata.get('n_features', len(feature_cols))}")
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
        "Upload a CSV of transactions to run fraud predictions. "
        "The model will assign a fraud probability to each row."
    )

    tab_predict, tab_dashboard, tab_models = st.tabs(
        ["Predict", "Dashboard", "Model Comparison"]
    )

    # ---- TAB: Predict ----
    with tab_predict:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            st.write(f"Uploaded **{raw_df.shape[0]:,}** rows, **{raw_df.shape[1]}** columns.")

            if not feature_cols:
                st.error("Could not determine expected feature columns from model metadata.")
                st.stop()

            aligned = align_features(raw_df, feature_cols)
            results = predict(model, aligned, feature_cols, threshold)

            n_flagged = int(results["is_flagged"].sum())
            n_total = len(results)
            flag_rate = n_flagged / n_total if n_total else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{n_total:,}")
            col2.metric("Flagged as Fraud", f"{n_flagged:,}")
            col3.metric("Flag Rate", f"{flag_rate:.2%}")

            st.markdown("---")
            st.subheader("Flagged Transactions")
            flagged = results[results["is_flagged"] == 1].sort_values(
                "fraud_probability", ascending=False
            )
            if flagged.empty:
                st.info("No transactions flagged at the current threshold.")
            else:
                st.dataframe(flagged.head(200), use_container_width=True)

            st.subheader("All Transactions")
            st.dataframe(
                results.sort_values("fraud_probability", ascending=False).head(500),
                use_container_width=True,
            )

            # Download
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("Upload a CSV file to start.")

    # ---- TAB: Dashboard ----
    with tab_dashboard:
        if uploaded is not None and "results" in dir():
            import matplotlib.pyplot as plt

            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                results["fraud_probability"],
                bins=50,
                edgecolor="black",
                alpha=0.7,
            )
            ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame(
                    {"feature": feature_cols, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=True).tail(15)

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

        # Show validation metrics from metadata
        val_metrics = metadata.get("val_metrics", metadata.get("metrics", {}).get("val", {}))
        test_metrics = metadata.get("test_metrics", metadata.get("metrics", {}).get("test", {}))

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
