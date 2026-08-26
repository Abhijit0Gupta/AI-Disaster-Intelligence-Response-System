from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
# BASELINE MODEL TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("BASELINE MODEL TRAINING")
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
# 2. DEFINE MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ------------------------------------------------------------
# 3. TRAIN AND EVALUATE
# ------------------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model.fit(X_train, y_train)

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

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

    # Save model
    model_filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(model, model_path)

    print("\nModel saved:")
    print(model_path)

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
# 5. SAVE RESULTS
# ------------------------------------------------------------

results_path = MODEL_DIR / "baseline_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)

# ------------------------------------------------------------
# 6. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("➡ Model analysis and optimization")

print("=" * 70)