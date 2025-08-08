import pandas as pd
import streamlit as st

def show_metrics(metrics):
    """Display metrics in nice cards."""
    if not metrics:
        st.warning("No metrics found.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", metrics.get("model", "?"))
    col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    col3.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    # Optional: classification report table
    if "classification_report" in metrics:
        report_df = pd.DataFrame(metrics["classification_report"]).T
        with st.expander("Classification Report"):
            st.dataframe(report_df)
