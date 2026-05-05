import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
)

# ----------------------------------------------------------------------
# Environment / project setup
# ----------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from contracts.settings import get_settings

# ----------------------------------------------------------------------
# 1.  Load and preprocess
# ----------------------------------------------------------------------
DATA_PATH = Path("data/bank-additional-full.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

# Drop the leaky feature (duration is recorded after the call ends)
df = df.drop(columns=["duration"])

# Target: binary 0/1
df["y"] = df["y"].map({"yes": 1, "no": 0})

# pdays = 999 is a sentinel → flag and drop original
df["pdays_was_999"] = (df["pdays"] == 999).astype(int)
df = df.drop(columns=["pdays"])

# Separate X and y
X = df.drop(columns=["y"])
y = df["y"]

# Identify numeric vs categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ----------------------------------------------------------------------
# 2.  Stratified 60/20/20 split
# ----------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"Train positive rate: {y_train.mean():.3f}")
print(f"Val   positive rate: {y_val.mean():.3f}")
print(f"Test  positive rate: {y_test.mean():.3f}")

# ----------------------------------------------------------------------
# 3.  Preprocessing pipeline (shared by all classifiers)
# ----------------------------------------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ----------------------------------------------------------------------
# 4.  Define candidate classifiers
# ----------------------------------------------------------------------
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42 , class_weight='balanced'),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1 ,
        class_weight='balanced'
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42 ),
}

# ----------------------------------------------------------------------
# 5.  Train each model, evaluate on validation set (AUC), pick the best
# ----------------------------------------------------------------------
best_model_name = None
best_auc = -1.0
best_pipeline = None
val_proba_best = None

for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"{name} validation AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_pipeline = pipeline
        val_proba_best = proba

print(f"\nBest model: {best_model_name} (validation AUC = {best_auc:.4f})")

# ----------------------------------------------------------------------
# 6.  Threshold tuning – EXACT instructor method from week 5 day 2
# ----------------------------------------------------------------------
precisions, recalls, thresholds_pr = precision_recall_curve(y_val, val_proba_best)

# F1 at every threshold (ignoring the last point where recall=0, precision=1)
f1s = (2 * precisions[:-1] * recalls[:-1]) / np.where(
    (precisions[:-1] + recalls[:-1]) > 0,
    precisions[:-1] + recalls[:-1],
    1.0,
)
best_f1_idx = int(np.argmax(f1s))
threshold_f1 = float(thresholds_pr[best_f1_idx])

# Highest threshold with recall >= 0.75
ok_recall_mask = recalls[:-1] >= 0.75
if ok_recall_mask.any():
    valid_thresholds = thresholds_pr[ok_recall_mask]
    threshold_recall075 = float(valid_thresholds.max())
else:
    threshold_recall075 = float(thresholds_pr[best_f1_idx])

print(f"\nThreshold maximising val F1: {threshold_f1:.4f}  (F1={f1s[best_f1_idx]:.4f})")
print(f"Threshold for val recall ≥ 0.75: {threshold_recall075:.4f}")

OPERATING_THRESHOLD = round(threshold_recall075, 4)
print(f"\nChosen OPERATING_THRESHOLD = {OPERATING_THRESHOLD}")

# Apply threshold
val_pred = (val_proba_best >= OPERATING_THRESHOLD).astype(int)
val_recall = recall_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred)
print(f"Validation metrics at threshold {OPERATING_THRESHOLD}:")
print(f"  Recall:    {val_recall:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  F1:        {val_f1:.4f}")

# ----------------------------------------------------------------------
# 7.  Evaluate best model on test set
# ----------------------------------------------------------------------
test_proba = best_pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= OPERATING_THRESHOLD).astype(int)

test_auc = roc_auc_score(y_test, test_proba)
test_f1 = f1_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)

print(f"\nTest metrics:")
print(f"  AUC:       {test_auc:.4f}")
print(f"  F1:        {test_f1:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  Precision: {test_precision:.4f}")

# ----------------------------------------------------------------------
# 8.  Confusion matrices and overfitting check
# ----------------------------------------------------------------------
train_proba = best_pipeline.predict_proba(X_train)[:, 1]
train_pred = (train_proba >= OPERATING_THRESHOLD).astype(int)

cm_train = confusion_matrix(y_train, train_pred)
cm_val = confusion_matrix(y_val, val_pred)
cm_test = confusion_matrix(y_test, test_pred)

print("\nConfusion Matrix (Train):\n", cm_train)
print("Confusion Matrix (Val):\n", cm_val)
print("Confusion Matrix (Test):\n", cm_test)

train_auc = roc_auc_score(y_train, train_proba)
val_auc = best_auc

print(f"\nOverfitting check (AUC):")
print(f"  Train AUC:  {train_auc:.4f}")
print(f"  Val AUC:    {val_auc:.4f}")
print(f"  Test AUC:   {test_auc:.4f}")
if train_auc - val_auc > 0.03:
    print("  ⚠️ Model may be overfitting (train-val gap > 0.03)")
else:
    print("  ✅ Overfitting not detected")

# ----------------------------------------------------------------------
# 9.  Save complete artifact locally (for direct joblib loading)
# ----------------------------------------------------------------------
model_artifact = {
    "pipeline": best_pipeline,
    "threshold": OPERATING_THRESHOLD,
    "model_name": best_model_name,
    "feature_names": X.columns.tolist(),
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target_recall_rule": 0.75,
}
joblib.dump(model_artifact, MODEL_DIR / "model.pkl")
print(f"\nModel saved to {MODEL_DIR / 'model.pkl'}")

# ----------------------------------------------------------------------
# 13. Compute and save reference distributions for drift detection
# ----------------------------------------------------------------------
print("\nComputing reference distributions for drift detection...")

# Numeric features – bin edges using 10 equal-frequency bins from train set
numeric_ref = {}
for col in numeric_cols:
    train_col = X_train[col].astype(float)
    # Create 10 bins based on quantiles
    bin_edges = np.quantile(train_col, q=np.linspace(0, 1, 11))
    # Ensure unique edges (some features may have many identical values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        # fallback: use linear bins if quantiles collapse
        bin_edges = np.linspace(train_col.min(), train_col.max(), 11)
    # Compute reference proportions for these bins on train set
    counts, _ = np.histogram(train_col, bins=bin_edges)
    ref_proportions = counts / counts.sum()
    numeric_ref[col] = {
        "bin_edges": bin_edges.tolist(),
        "ref_proportions": ref_proportions.tolist()
    }

# Categorical features – frequency of each category
categorical_ref = {}
for col in categorical_cols:
    train_col = X_train[col].astype(str)
    counts = train_col.value_counts()
    # Keep all categories seen in training; any new category later gets tiny frequency
    total = counts.sum()
    ref_proportions = (counts / total).to_dict()
    categorical_ref[col] = {
        "categories": list(ref_proportions.keys()),
        "ref_proportions": ref_proportions
    }

# Output (predicted probability) distribution on training set
train_proba_ref = best_pipeline.predict_proba(X_train)[:, 1]
output_bin_edges = np.linspace(0, 1, 11)
output_counts, _ = np.histogram(train_proba_ref, bins=output_bin_edges)
output_ref_proportions = output_counts / output_counts.sum()
output_ref = {
    "bin_edges": output_bin_edges.tolist(),
    "ref_proportions": output_ref_proportions.tolist()
}

reference = {
    "numeric": numeric_ref,
    "categorical": categorical_ref,
    "output": output_ref
}

REF_PATH = MODEL_DIR / "reference.json"
with open(REF_PATH, "w") as f:
    json.dump(reference, f, indent=2)
print(f"Reference distributions saved to {REF_PATH}")

# ----------------------------------------------------------------------
# 10.  Helper to plot confusion matrix (pure matplotlib – no seaborn)
# ----------------------------------------------------------------------
def plot_confusion_matrix(cm, title):
    """Save a confusion matrix as a PNG using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    buf = "cm.png"
    plt.savefig(buf)
    plt.close()
    return buf

cm_test_img = plot_confusion_matrix(cm_test, f"Test Confusion Matrix (t={OPERATING_THRESHOLD})")

# ----------------------------------------------------------------------
# 11.  Custom pyfunc wrapper for MLflow serving
# ----------------------------------------------------------------------
class ThresholdClassifierWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper that applies a pre‑computed decision threshold."""

    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        proba = self.pipeline.predict_proba(model_input)[:, 1]
        return (proba >= self.threshold).astype(int)

# ----------------------------------------------------------------------
# 12.  MLflow registration
# ----------------------------------------------------------------------
settings = get_settings()

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("bank_marketing_experiment")

wrapped_model = ThresholdClassifierWrapper(best_pipeline, OPERATING_THRESHOLD)

input_example = X_val.head(1)
signature = infer_signature(X_val, wrapped_model.predict(None, X_val))

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("classifier", best_model_name)
    mlflow.log_param("target_recall", 0.75)
    mlflow.log_param("chosen_threshold", OPERATING_THRESHOLD)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("random_state", 42)
    mlflow.log_param("best_model_selection_metric", "validation_AUC")
    for name in classifiers:
        mlflow.log_param(f"model_{name}", "trained")

    # Log metrics
    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("val_recall", val_recall)

    # Log confusion matrix artifact
    mlflow.log_artifact(cm_test_img)
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="sklearn_model",
        registered_model_name="bank_marketing_classifier",
        input_example=input_example,
        signature=signature,
    )
    # --- NEW: store threshold & feature names as tags ---
    mlflow.set_tag("threshold", OPERATING_THRESHOLD)
    mlflow.set_tag("feature_names", json.dumps(X.columns.tolist()))
    mlflow.set_tag("model_name", best_model_name)

    # Log the model
    mlflow.pyfunc.log_model(
        name="model",
        python_model=wrapped_model,
        registered_model_name="bank_marketing_classifier",
        input_example=input_example,
        signature=signature,
    )

    # Schema artifact
    schema = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target": "y",
        "preprocessing": "StandardScaler + OneHotEncoder",
        "dropped_features": ["duration"],
        "pdays_sentinel": "flag pdays_was_999 added, original dropped",
        "unknown_treatment": "retained as category",
    }
    with open("schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    mlflow.log_artifact("schema.json")

    # Model card
    model_card = (
        f"Binary classifier for bank marketing dataset.\n"
        f"Best model selected by validation AUC: {best_model_name}.\n"
        f"Trained on {len(X_train)} samples, validated on {len(X_val)}.\n"
        f"Threshold tuned using week 5 day 2 rule (highest threshold with recall ≥ 0.75): {OPERATING_THRESHOLD}.\n"
        f"Train AUC = {train_auc:.4f}, Val AUC = {val_auc:.4f}, Test AUC = {test_auc:.4f}.\n"
        f"Test F1 = {test_f1:.4f}, Recall = {test_recall:.4f}, Precision = {test_precision:.4f}.\n"
        f"Feature count: {len(X.columns)}. \n"
        f"Overfitting gap (train-val AUC) = {train_auc - val_auc:.4f}.\n"
    )
    with open("model_card.md", "w") as f:
        f.write(model_card)
    mlflow.log_artifact("model_card.md")

    # Environment fingerprint
    env_info = {
        "python_version": sys.version,
        "sklearn_version": __import__("sklearn").__version__,
        "pandas_version": pd.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open("environment.json", "w") as f:
        json.dump(env_info, f, indent=2)
    mlflow.log_artifact("environment.json")

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run ID: {run_id}")
    print(f"Model registered as 'bank_marketing_classifier' (best: {best_model_name})")