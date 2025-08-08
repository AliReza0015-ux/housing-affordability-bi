# model.py
import joblib
import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path("outputs")
MODEL_DIR = Path("models")

def list_models():
    return [
        "random_forest" if (ARTIFACT_DIR / "random_forest_metrics.json").exists() else None,
        "xgboost" if (ARTIFACT_DIR / "xgboost_metrics.json").exists() else None
    ]

def load_metrics(model_name):
    path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    return None if not path.exists() else pd.read_json(path).to_dict()  # not used by upload, but kept

def load_metrics_raw(model_name):
    import json
    path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    if not path.exists(): return None
    with open(path) as f: return json.load(f)

def load_confusion_matrix_img(model_name):
    p = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
    return str(p) if p.exists() else None

def load_feature_importance_img(model_name):
    p = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.png"
    return str(p) if p.exists() else None

def load_feature_importance_table(model_name):
    p = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.csv"
    return None if not p.exists() else pd.read_csv(p)

def load_model(model_name):
    p = MODEL_DIR / f"{model_name}.pkl"
    return joblib.load(p)

def get_feature_list():
    p = ARTIFACT_DIR / "features_for_modeling.csv"
    s = pd.read_csv(p)["feature"].tolist()
    return s  # order matters

def prepare_features(df: pd.DataFrame, feature_list):
    # keep only expected features, in order; fill missing with 0
    cols = {c: (df[c] if c in df.columns else 0) for c in feature_list}
    X = pd.DataFrame(cols)[feature_list]
    # try to coerce to numeric where possible
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="ignore")
    return X

def predict_on_df(df: pd.DataFrame, model_name: str):
    feature_list = get_feature_list()
    model = load_model(model_name)
    X = prepare_features(df, feature_list)
    preds = model.predict(X)
    return X, preds
