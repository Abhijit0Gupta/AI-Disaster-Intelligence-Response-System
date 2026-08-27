from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# MODEL ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("MODEL ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD TEST DATA
# ------------------------------------------------------------

X_test = np.load(
    MODEL_DIR / "X_test_processed.npy"
)

y_test = pd.read_csv(
    MODEL_DIR / "y_test.csv"
).squeeze()

print("\nTest data shape:", X_test.shape)
print("Test labels   :", len(y_test))

# ------------------------------------------------------------
# 2. LOAD TRAINED MODELS
# ------------------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

models = {}

print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)

for name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    models[name] = joblib.load(model_path)

    print(f"✅ {name} loaded")

# ------------------------------------------------------------
# 3. MODEL ANALYSIS
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 70)

for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"MODEL: {name}")
    print("-" * 70)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nInterpretation:")

    tn, fp, fn, tp = cm.ravel()

    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "False_Negatives": fn,
        "False_Positives": fp
    })

# ------------------------------------------------------------
# 4. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# 5. BEST MODELS
# ------------------------------------------------------------

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]

best_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
]

best_roc_auc = results_df.loc[
    results_df["ROC_AUC"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL BY METRIC")
print("=" * 70)

print(
    "\nBest Accuracy:"
    f" {best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    "Best Recall:"
    f" {best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    "Best F1 Score:"
    f" {best_f1['Model']} "
    f"({best_f1['F1']:.4f})"
)

print(
    "Best ROC-AUC:"
    f" {best_roc_auc['Model']} "
    f"({best_roc_auc['ROC_AUC']:.4f})"
)

# ------------------------------------------------------------
# 6. FLOOD DETECTION PRIORITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FLOOD DETECTION PRIORITY")
print("=" * 70)

print(
    "\nFor disaster prediction, recall is highly important."
)

print(
    "A false negative means an actual flood was missed."
)

print(
    "\nModel ranking by Recall:"
)

recall_ranking = results_df.sort_values(
    "Recall",
    ascending=False
)

for _, row in recall_ranking.iterrows():

    print(
        f"{row['Model']}: "
        f"Recall={row['Recall']:.4f}, "
        f"False Negatives={int(row['False_Negatives'])}"
    )

# ------------------------------------------------------------
# 7. SAVE ANALYSIS
# ------------------------------------------------------------

analysis_path = MODEL_DIR / "model_analysis.csv"

results_df.to_csv(
    analysis_path,
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS SAVED")
print("=" * 70)

print("\nOutput file:")
print(analysis_path)

print("\n" + "=" * 70)
print("MODEL ANALYSIS COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("➡ Model optimization and tuning")

print("=" * 70)