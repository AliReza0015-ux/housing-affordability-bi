# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

import model
import utils

st.set_page_config(page_title=" Housing Affordability Dashboard", layout="wide")
st.title(" Housing Affordability – Model Dashboard")

# -------------------------
# Cached loaders for performance
# -------------------------
@st.cache_resource
def load_model_cached(name: str):
    return model.load_model(name)

@st.cache_data
def load_metrics_cached(name: str):
    return model.load_metrics_raw(name)

# -------------------------
# Sidebar – model picker
# -------------------------
st.sidebar.title("Models")
models = [m for m in model.list_models() if m]
if not models:
    st.error("No artifacts in outputs/. Run your Phase-2 notebook freeze cells first.")
    st.stop()

chosen = st.sidebar.radio("Choose a model", models)

with st.sidebar.expander("📄 Download feature template"):
    # let users grab the exact schema your model expects
    utils.schema_template_download(model.get_feature_list())

# -------------------------
# Metrics & visuals
# -------------------------
raw_metrics = load_metrics_cached(chosen) or {}

# show only the key KPIs as cards
cards = {
    "model": raw_metrics.get("model", "?"),
    "accuracy": raw_metrics.get("accuracy", 0.0),
}
cr = raw_metrics.get("classification_report", {})
f1w = cr.get("weighted avg", {}).get("f1-score")
if f1w is not None:
    cards["f1 (weighted)"] = f1w

utils.show_metrics(cards)

conf_img = model.load_confusion_matrix_img(chosen)
fi_img = model.load_feature_importance_img(chosen)
fi_tbl = model.load_feature_importance_table(chosen)

cols = st.columns(2)
if conf_img:
    cols[0].image(conf_img, caption=f"{chosen} – Confusion Matrix", use_container_width=True)
if fi_img:
    cols[1].image(fi_img, caption=f"{chosen} – Top 5 Features", use_container_width=True)
if fi_tbl is not None:
    with st.expander("Feature Importance (Top 5) – Table"):
        st.dataframe(fi_tbl, use_container_width=True)

st.markdown("---")

# -------------------------
# Upload → Predict → Download
# -------------------------
st.subheader(" Batch Predictions")

uploaded = st.file_uploader(
    "Upload a CSV to score (must include the training features in any order)",
    type=["csv"]
)

if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("**Preview of uploaded data:**")
        st.dataframe(df_in.head(), use_container_width=True)

        # Schema validation
        feature_list = model.get_feature_list()
        missing, extra = utils.check_schema(df_in, feature_list)
        utils.display_schema_warnings(missing, extra)
        if missing:
            st.stop()

        # keep only the required columns, in the correct order
        df_in = df_in[feature_list]

        # Predict
        mdl = load_model_cached(chosen)
        _, preds = model.predict_on_df(df_in, chosen)
        result = df_in.copy()
        result[f"prediction_{chosen}"] = preds

        st.success(f"Predictions generated with {chosen}.")
        st.dataframe(result.head(), use_container_width=True)

        # Download predictions
        utils.df_to_csv_download(
            result,
            filename=f"predictions_{chosen}.csv",
            label="⬇Download predictions CSV"
        )

        with st.expander("Columns used for prediction (ordered)"):
            st.code("\n".join(feature_list))

    except Exception as e:
        st.error(f"Failed to score file: {e}")
else:
    st.info("Upload a CSV to run predictions. You can download the feature template from the sidebar.")
