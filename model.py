# model.py
from pathlib import Path
import json
import joblib
import pandas as pd

ARTIFACT_DIR = Path("outputs")
MODEL_DIR = Path("models")

# ---------- Discovery ----------

def list_models():
    """Return list of model names present based on metrics files in outputs/."""
    names = []
    if (ARTIFACT_DIR / "random_forest_metrics.json").exists():
        names.append("random_forest")
    if (ARTIFACT_DIR / "xgboost_metrics.json").exists():
        names.append("xgboost")
    return names

# ---------- Metrics & Artifacts ----------

def load_metrics_raw(model_name: str):
    """Load metrics JSON (dict) for the given model name."""
    path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def load_confusion_matrix_img(model_name: str):
    """Return string path to confusion matrix PNG, or None."""
    p = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
    return str(p) if p.exists() else None

def load_feature_importance_img(model_name: str):
    """Return string path to top-5 feature importance PNG, or None."""
    p = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.png"
    return str(p) if p.exists() else None

def load_feature_importance_table(model_name: str):
    """Return DataFrame of top-5 feature importances, or None."""
    p = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.csv"
    return pd.read_csv(p) if p.exists() else None

# ---------- Models ----------

def _model_path(model_name: str) -> Path:
    """Map logical model name → pkl path."""
    name_to_file = {
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl",
    }
    return MODEL_DIR / name_to_file[model_name]

def load_model(model_name: str):
    """Load and return the trained model object for given name."""
    p = _model_path(model_name)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)

# ---------- Features & Prediction ----------

def get_feature_list():
    """
    Return the exact ordered feature list used during training
    (saved by your notebook freeze step).
    """
    path = ARTIFACT_DIR / "features_for_modeling.csv"
    if not path.exists():
        raise FileNotFoundError("outputs/features_for_modeling.csv not found. "
                                "Run the notebook freeze cells first.")
    return pd.read_csv(path)["feature"].tolist()

def prepare_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Align incoming dataframe to the training feature order.
    - Keep only expected columns, in order.
    - Fill missing features with 0 (safe default for numeric; adjust if you prefer).
    - Coerce obvious numerics where possible (non-destructive for strings).
    """
    aligned = {}
    for col in feature_list:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = 0  # default fill; change to NaN if you prefer strict fail
    X = pd.DataFrame(aligned)[feature_list]

    # Try numeric coercion (won't break valid strings)
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="ignore")
    return X

def predict_on_df(df: pd.DataFrame, model_name: str):
    """
    Prepare incoming df to the training schema and predict with the selected model.
    Returns (X_used, predictions)
    """
    feature_list = get_feature_list()
    mdl = load_model(model_name)
    X = prepare_features(df, feature_list)
    preds = mdl.predict(X)
    return X, preds

# Permutation importance assets (optional)
def load_perm_importance_img(model_name: str):
    """Return path to permutation importance PNG if it exists."""
    p = ARTIFACT_DIR / f"{model_name}_permutation_importance_top10.png"
    return str(p) if p.exists() else None

def load_perm_importance_table(model_name: str):
    """Return DataFrame of permutation importance (top 10) if it exists."""
    p = ARTIFACT_DIR / f"{model_name}_permutation_importance_top10.csv"
    return pd.read_csv(p) if p.exists() else None

