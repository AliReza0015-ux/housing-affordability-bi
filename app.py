# app.py
import streamlit as st
import pandas as pd
import model
import utils
from pathlib import Path

st.set_page_config(page_title="Housing Affordability Dashboard", layout="wide")
st.title("Housing Affordability – Model Dashboard")

# -------------------------
# Cache loaders for performance
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

# -------------------------
# Metrics & visuals
# -------------------------
metrics = load_metrics_cached(chosen)
utils.show_metrics(metrics)

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
st.subheader("Batch Predictions")
uploaded = st.file_uploader(
    "Upload a CSV to score (must include the training features)",
    type=["csv"]
)

if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("**Preview of uploaded data:**")
        st.dataframe(df_in.head(), use_container_width=True)

        # Schema validation
        feature_list = model.get_feature_list()
        missing = [col for col in feature_list if col not in df_in.columns]
        extra = [col for col in df_in.columns if col not in feature_list]

        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        if extra:
            st.warning(f"Extra columns ignored: {extra}")
            df_in = df_in[feature_list]

        # Predictions
        model_obj = load_model_cached(chosen)
        _, preds = model.predict_on_df(df_in, chosen)
        result = df_in.copy()
        result[f"prediction_{chosen}"] = preds

        st.success(f"Predictions generated with {chosen}.")
        st.dataframe(result.head(), use_container_width=True)

        utils.df_to_csv_download(result,
                                 filename=f"predictions_{chosen}.csv",
                                 label="⬇️ Download predictions CSV")
        with st.expander("Columns used for prediction (in order)"):
            st.code("\n".join(feature_list))
    except Exception as e:
        st.error(f"Failed to score file: {e}")
else:
    st.info("Upload a CSV to run predictions. The app will align columns to the training feature list and fill any missing with 0.")
