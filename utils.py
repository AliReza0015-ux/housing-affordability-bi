# utils.py
import io
import pandas as pd
import streamlit as st

def show_metrics(metrics):
    if not metrics: 
        st.warning("No metrics found.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", metrics.get("model", "?"))
    col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    # optional f1 if present
    f1 = metrics.get("classification_report", {}).get("weighted avg", {}).get("f1-score")
    if f1 is not None:
        col3.metric("F1 (weighted)", f"{f1:.3f}")

    if "classification_report" in metrics:
        report_df = pd.DataFrame(metrics["classification_report"]).T
        with st.expander("Classification Report"):
            st.dataframe(report_df)

def df_to_csv_download(df: pd.DataFrame, filename: str, label="Download CSV"):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv"
    )
