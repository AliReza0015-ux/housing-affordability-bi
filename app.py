# app.py
import streamlit as st
import pandas as pd
import model
import utils

st.set_page_config(page_title="🏠 Housing Affordability Dashboard", layout="wide")
st.title("🏠 Housing Affordability – Model Dashboard")

# Sidebar – model picker
st.sidebar.title("Models")
models = [m for m in model.list_models() if m]
if not models:
    st.error("No artifacts in outputs/. Run your Phase-2 notebook freeze cells first.")
    st.stop()
chosen = st.sidebar.radio("Choose a model", models)

# ---- Metrics & visuals (existing section) ----
metrics = model.load_metrics_raw(chosen)
utils.show_metrics(metrics)

conf_img = model.load_confusion_matrix_img(chosen)
fi_img = model.load_feature_importance_img(chosen)
fi_tbl = model.load_feature_importance_table(chosen)

cols = st.columns(2)
if conf_img: cols[0].image(conf_img, caption=f"{chosen} – Confusion Matrix", use_column_width=True)
if fi_img:   cols[1].image(fi_img, caption=f"{chosen} – Top 5 Features", use_column_width=True)
if fi_tbl is not None:
    with st.expander("Feature Importance (Top 5) – Table"):
        st.dataframe(fi_tbl)

st.markdown("---")

# ---- Upload → Predict → Download ----
st.subheader("🔄 Batch Predictions")
uploaded = st.file_uploader("Upload a CSV to score (must include the training features)", type=["csv"])
if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("**Preview of uploaded data:**")
        st.dataframe(df_in.head())

        X_used, preds = model.predict_on_df(df_in, chosen)
        result = df_in.copy()
        result[f"prediction_{chosen}"] = preds

        st.success(f"Predictions generated with {chosen}.")
        st.dataframe(result.head())

        utils.df_to_csv_download(result, filename=f"predictions_{chosen}.csv", label="⬇️ Download predictions CSV")
        with st.expander("Columns used for prediction (in order)"):
            st.code("\n".join(model.get_feature_list()))
    except Exception as e:
        st.error(f"Failed to score file: {e}")
else:
    st.info("Upload a CSV to run predictions. The app will align columns to the training feature list and fill any missing with 0.")
