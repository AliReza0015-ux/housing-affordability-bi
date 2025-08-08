import streamlit as st
import model
import utils

st.set_page_config(page_title=" Housing Affordability Dashboard", layout="wide")
st.title("Housing Affordability – Model Dashboard")

# Sidebar
st.sidebar.title("Select Model")
models = [m for m in model.list_models() if m]
if not models:
    st.error("No models found in outputs/. Please run your Phase-2 notebook first.")
    st.stop()

selected_model = st.sidebar.radio("Choose a model:", models)

# Load and display
metrics = model.load_metrics(selected_model)
utils.show_metrics(metrics)

conf_img = model.load_confusion_matrix_img(selected_model)
if conf_img:
    st.image(conf_img, caption=f"{selected_model} – Confusion Matrix")

fi_img = model.load_feature_importance_img(selected_model)
fi_table = model.load_feature_importance_table(selected_model)

if fi_img or fi_table is not None:
    st.subheader("Top 5 Features")
    cols = st.columns(2)
    if fi_img:
        cols[0].image(fi_img, caption="Feature Importance (Top 5)")
    if fi_table is not None:
        cols[1].dataframe(fi_table)
