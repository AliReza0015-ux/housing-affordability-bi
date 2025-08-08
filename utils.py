# utils.py
import streamlit as st
import pandas as pd
import io

# ------------------------------
# Metrics Display
# ------------------------------
def show_metrics(metrics: dict):
    """
    Display model metrics in a nice horizontal layout.
    Expects keys like 'accuracy', 'precision', 'recall', 'f1'.
    """
    if not metrics:
        st.warning("⚠️ No metrics available for this model.")
        return
    st.subheader("📊 Model Performance")
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(label=k.capitalize(), value=f"{v:.3f}" if isinstance(v, (int, float)) else str(v))


# ------------------------------
# CSV Download Helpers
# ------------------------------
def df_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    """
    Add a download button to export a DataFrame as CSV.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )


def schema_template_download(feature_list: list, filename="template_features.csv"):
    """
    Allow users to download a CSV template with correct headers.
    """
    template_df = pd.DataFrame(columns=feature_list)
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇Download CSV Template",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )


# ------------------------------
# Column Check Helpers
# ------------------------------
def check_schema(df: pd.DataFrame, required_features: list):
    """
    Compare uploaded DataFrame with the required feature list.
    Returns (missing_cols, extra_cols)
    """
    missing = [col for col in required_features if col not in df.columns]
    extra = [col for col in df.columns if col not in required_features]
    return missing, extra


def display_schema_warnings(missing, extra):
    """
    Display missing/extra column warnings in Streamlit.
    """
    if missing:
        st.error(f"Missing required columns: {missing}")
    if extra:
        st.warning(f"Extra columns ignored: {extra}")
