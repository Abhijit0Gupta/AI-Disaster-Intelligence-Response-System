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
    confusion_matrix,
    classification_report
)

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# FINAL MODEL SELECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("FINAL MODEL SELECTION")
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
# 2. LOAD MODELS
# ------------------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Optimized Random Forest": "random_forest_optimized.pkl"
}

models = {}

print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

for name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    if model_path.exists():
        models[name] = joblib.load(model_path)
        print(f"✅ {name} loaded")
    else:
        print(f"⚠️ {name} not found")

# ------------------------------------------------------------
# 3. EVALUATE ALL MODELS
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"MODEL: {name}")
    print("-" * 70)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    tn, fp, fn, tp = cm.ravel()

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nFalse Negatives:", fn)
    print("False Positives:", fp)

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
# 4. CREATE COMPARISON TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL RANKING")
print("=" * 70)

ranking = results_df.sort_values(
    by=["Recall", "F1", "ROC_AUC", "Accuracy"],
    ascending=False
)

print(
    ranking.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# 5. SELECT FINAL MODEL
# ------------------------------------------------------------

final_model_name = ranking.iloc[0]["Model"]

final_model = models[
    final_model_name
]

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)

print(
    f"\nSelected model: {final_model_name}"
)

# ------------------------------------------------------------
# 6. FINAL MODEL DETAILS
# ------------------------------------------------------------

final_row = ranking.iloc[0]

print("\nFinal model performance:")

print(
    f"Accuracy : {final_row['Accuracy']:.4f}"
)

print(
    f"Precision: {final_row['Precision']:.4f}"
)

print(
    f"Recall   : {final_row['Recall']:.4f}"
)

print(
    f"F1 Score : {final_row['F1']:.4f}"
)

print(
    f"ROC-AUC  : {final_row['ROC_AUC']:.4f}"
)

print(
    f"False Negatives: "
    f"{int(final_row['False_Negatives'])}"
)

print(
    f"False Positives: "
    f"{int(final_row['False_Positives'])}"
)

# ------------------------------------------------------------
# 7. SAVE FINAL MODEL
# ------------------------------------------------------------

final_model_path = (
    MODEL_DIR / "final_flood_model.pkl"
)

joblib.dump(
    final_model,
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)

# ------------------------------------------------------------
# 8. SAVE MODEL COMPARISON
# ------------------------------------------------------------

comparison_path = (
    MODEL_DIR / "final_model_comparison.csv"
)

ranking.to_csv(
    comparison_path,
    index=False
)

print("\nModel comparison saved:")
print(comparison_path)

# ------------------------------------------------------------
# 9. FINAL CLASSIFICATION REPORT
# ------------------------------------------------------------

y_final_pred = final_model.predict(
    X_test
)

print("\n" + "=" * 70)
print("FINAL MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_final_pred,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 10. PROJECT DECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROJECT DECISION")
print("=" * 70)

print(
    "\nThe final model prioritizes flood detection recall."
)

print(
    "This is appropriate for a disaster-response system "
    "where missing an actual flood can be more serious "
    "than generating a false alarm."
)

print(
    f"\nFINAL MODEL: {final_model_name}"
)

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("➡ Final model validation and prediction pipeline")

print("=" * 70)