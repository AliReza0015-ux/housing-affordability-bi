import json
from pathlib import Path
import pandas as pd

ARTIFACT_DIR = Path("outputs")

def list_models():
    """Detect available models based on existing metrics files."""
    return [
        "random_forest" if (ARTIFACT_DIR / "random_forest_metrics.json").exists() else None,
        "xgboost" if (ARTIFACT_DIR / "xgboost_metrics.json").exists() else None
    ]

def load_metrics(model_name):
    """Load metrics JSON for a given model."""
    path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_confusion_matrix_img(model_name):
    """Return path to confusion matrix image."""
    path = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
    return str(path) if path.exists() else None

def load_feature_importance_img(model_name):
    """Return path to top 5 feature importance image."""
    path = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.png"
    return str(path) if path.exists() else None

def load_feature_importance_table(model_name):
    """Load top 5 feature importance CSV."""
    path = ARTIFACT_DIR / f"{model_name}_feature_importance_top5.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
