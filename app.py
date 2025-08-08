# app.py
import streamlit as st
import pandas as pd

import model
import utils

st.set_page_config(page_title="🏠 Housing Affordability Dashboard", layout="wide")
st.title("🏠 Housing Affordability – Model Dashboard")

# -------------------------
# Cached loaders for performance
# -------------------------
@st.cache_resource
def load_model_cached(name: str):
    return model.load_model(name)

@st.cache_data
def load_metrics_cached(name: str):
    return model.load_metrics_raw(name)

@st.cache_data
def load_cv_summary():
    import json, os
    p = "outputs/cv_summary.json"
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)

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
    utils.schema_template_download(model.get_feature_list())

# -------------------------
# Metrics & visuals
# -------------------------
raw_metrics = load_metrics_cached(chosen) or {}

# KPI cards
cards = {
    "model": raw_metrics.get("model", "?"),
    "accuracy": raw_metrics.get("accuracy", 0.0),
}
cr = raw_metrics.get("classification_report", {})
f1w = cr.get("weighted avg", {}).get("f1-score")
if f1w is not None:
    cards["f1 (weighted)"] = f1w
utils.show_metrics(cards)

# CV summary (if you saved outputs/cv_summary.json from the notebook)
cv = load_cv_summary()
if cv:
    cols = st.columns(2)
    if cv.get("rf"):
        cols[0].info(f"RF CV acc: **{cv['rf']['mean']:.3f} ± {cv['rf']['std']:.3f}** (5-fold)")
    if cv.get("xgb"):
        cols[1].info(f"XGB CV acc: **{cv['xgb']['mean']:.3f} ± {cv['xgb']['std']:.3f}** (5-fold)")

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

# ---- Interpretability: Permutation Importance ----
# NOTE: If your notebook saved *top10* files, either change your notebook to top5,
# or change the caption below to say "Top 10".
pi_img = model.load_perm_importance_img(chosen)
pi_tbl = model.load_perm_importance_table(chosen)

with st.expander("🧠 Interpretability – Permutation Importance (Top 5)"):
    if not pi_img and pi_tbl is None:
        st.info("Permutation importance artifacts not found. Generate them in the notebook to enable this section.")
    else:
        cols_pi = st.columns(2)
        if pi_img:
            cols_pi[0].image(pi_img, caption=f"{chosen} – Permutation Importance (Top 5)", use_container_width=True)
        if pi_tbl is not None:
            cols_pi[1].dataframe(pi_tbl, use_container_width=True)

st.markdown("---")

# -------------------------
# Upload → Predict → Download
# -------------------------
st.subheader("🔄 Batch Predictions")

uploaded = st.file_uploader(
    "Upload a CSV to score (must include the training features in any order)",
    type=["csv"]
)

if uploaded:
    try:
        # Friendly CSV read (handles utf-8-sig and commas)
        try:
            df_in = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df_in = pd.read_csv(uploaded, encoding="utf-8-sig")

        st.write("**Preview of uploaded data:**")
        st.dataframe(df_in.head(), use_container_width=True)

        # Schema validation
        feature_list = model.get_feature_list()
        missing, extra = utils.check_schema(df_in, feature_list)
        utils.display_schema_warnings(missing, extra)
        if missing:
            st.stop()

        # keep only required columns in correct order
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
