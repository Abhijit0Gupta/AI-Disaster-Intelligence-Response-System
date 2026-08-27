from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
# MODEL OPTIMIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("MODEL OPTIMIZATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD PROCESSED DATA
# ------------------------------------------------------------

X_train = np.load(
    MODEL_DIR / "X_train_processed.npy"
)

X_test = np.load(
    MODEL_DIR / "X_test_processed.npy"
)

y_train = pd.read_csv(
    MODEL_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    MODEL_DIR / "y_test.csv"
).squeeze()

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# ------------------------------------------------------------
# 2. DEFINE RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST OPTIMIZATION")
print("=" * 70)

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# 3. HYPERPARAMETER GRID
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

print("\nHyperparameter combinations:")
print("n_estimators :", param_grid["n_estimators"])
print("max_depth    :", param_grid["max_depth"])
print("min_samples_split:", param_grid["min_samples_split"])
print("min_samples_leaf :", param_grid["min_samples_leaf"])

total_combinations = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
)

print(
    "\nTotal combinations:",
    total_combinations
)

# ------------------------------------------------------------
# 4. GRID SEARCH
# ------------------------------------------------------------

print("\nStarting GridSearchCV...")
print("Scoring metric: Recall")
print("Cross-validation folds: 3")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)

print("\nGrid search complete.")

# ------------------------------------------------------------
# 5. BEST PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(
    "\nBest parameters:"
)

print(
    grid_search.best_params_
)

print(
    "\nBest cross-validation recall:",
    round(grid_search.best_score_, 4)
)

# ------------------------------------------------------------
# 6. BEST MODEL
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

# ------------------------------------------------------------
# 7. TEST SET EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OPTIMIZED MODEL TEST PERFORMANCE")
print("=" * 70)

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

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

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 8. COMPARE WITH BASELINE RANDOM FOREST
# ------------------------------------------------------------

baseline_path = MODEL_DIR / "baseline_results.csv"

if baseline_path.exists():

    baseline_df = pd.read_csv(
        baseline_path
    )

    baseline_rf = baseline_df[
        baseline_df["Model"] == "Random Forest"
    ]

    if not baseline_rf.empty:

        baseline_rf = baseline_rf.iloc[0]

        print("\n" + "=" * 70)
        print("BASELINE VS OPTIMIZED RANDOM FOREST")
        print("=" * 70)

        print(
            "\nMetric          Baseline      Optimized"
        )

        print(
            f"Accuracy        "
            f"{baseline_rf['Accuracy']:.4f}        "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision       "
            f"{baseline_rf['Precision']:.4f}        "
            f"{precision:.4f}"
        )

        print(
            f"Recall          "
            f"{baseline_rf['Recall']:.4f}        "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score        "
            f"{baseline_rf['F1']:.4f}        "
            f"{f1:.4f}"
        )

        print(
            f"ROC-AUC         "
            f"{baseline_rf['ROC_AUC']:.4f}        "
            f"{roc_auc:.4f}"
        )

# ------------------------------------------------------------
# 9. SAVE OPTIMIZED MODEL
# ------------------------------------------------------------

optimized_model_path = (
    MODEL_DIR / "random_forest_optimized.pkl"
)

joblib.dump(
    best_model,
    optimized_model_path
)

print("\n" + "=" * 70)
print("OPTIMIZED MODEL SAVED")
print("=" * 70)

print(
    "\nModel:",
    optimized_model_path
)

# ------------------------------------------------------------
# 10. SAVE OPTIMIZATION RESULTS
# ------------------------------------------------------------

optimization_results = pd.DataFrame([{
    "Model": "Random Forest Optimized",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc,
    "False_Negatives": fn,
    "False_Positives": fp,
    "Best_Parameters": str(
        grid_search.best_params_
    )
}])

results_path = (
    MODEL_DIR / "optimization_results.csv"
)

optimization_results.to_csv(
    results_path,
    index=False
)

print("\nOptimization results saved:")
print(results_path)

# ------------------------------------------------------------
# 11. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL OPTIMIZATION COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("➡ Final model selection and evaluation")

print("=" * 70)