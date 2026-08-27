from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import load

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "India_Flood_Inventory_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "india_flood_impact_random_forest.pkl"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "models"
    / "india_feature_importance.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

EVALUATION_FILE = (
    RESULTS_DIR
    / "india_evaluation_summary.csv"
)


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - MODEL EVALUATION")
print("=" * 70)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_FILE}"
    )

print("Feature dataset found.")
print("Trained model found.")


# ============================================================
# 5. LOAD FEATURE DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING FEATURE DATA")
print("=" * 70)

df = pd.read_csv(
    FEATURE_FILE
)

print("\nDataset loaded successfully!")

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)


# ============================================================
# 6. TARGET
# ============================================================

TARGET = "High_Impact"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(
    "\nTarget column:",
    TARGET
)

print(
    "\nTarget distribution:"
)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# ============================================================
# 8. REMOVE SAME COLUMNS AS TRAINING
# ============================================================

print("\n" + "=" * 70)
print("REMOVING NON-PREDICTIVE / HIGH-CARDINALITY COLUMNS")
print("=" * 70)

columns_to_drop = [
    "Event Source",
    "Districts"
]

for column in columns_to_drop:

    if column in X.columns:

        X = X.drop(
            columns=[column]
        )

        print(
            f"Removed: {column}"
        )


# ============================================================
# 9. LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

LEAKAGE_COLUMNS = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Extent of damage",
    "Total_Human_Impact",
    "Reported_Human_Impact"
]

leakage_found = []

for column in LEAKAGE_COLUMNS:

    if column in X.columns:

        leakage_found.append(
            column
        )

        X = X.drop(
            columns=[column]
        )

        print(
            f"EXCLUDED: {column}"
        )


if not leakage_found:

    print(
        "PASS: No known leakage columns found."
    )


# ============================================================
# 10. RECREATE SAME TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("RECREATING TEST SET")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)

print(
    "\nTesting target distribution:"
)

print(
    y_test
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = load(
    MODEL_FILE
)

print(
    "\nModel loaded successfully!"
)

print(
    "Model type:",
    type(model).__name__
)


# ============================================================
# 12. GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST SET PREDICTIONS")
print("=" * 70)

try:

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

except Exception as error:

    print(
        "\nERROR while generating predictions:"
    )

    print(error)

    raise


print(
    "\nTest predictions generated successfully!"
)


# ============================================================
# 13. CALCULATE PERFORMANCE
# ============================================================

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
    y_probability
)

average_precision = (
    average_precision_score(
        y_test,
        y_probability
    )
)


# ============================================================
# 14. PRINT PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("TEST SET MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print()

print(cm)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Indian Flood Impact Model"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    [
        "Predicted 0",
        "Predicted 1"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Actual 0",
        "Actual 1"
    ]
)

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()


CONFUSION_FILE = (
    FIGURES_DIR
    / "india_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved to:"
)

print(
    CONFUSION_FILE
)


# ============================================================
# 18. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Indian Flood Impact Model"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()


ROC_FILE = (
    FIGURES_DIR
    / "india_roc_curve.png"
)

plt.savefig(
    ROC_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "ROC curve saved to:"
)

print(
    ROC_FILE
)


# ============================================================
# 19. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_values,
    precision_values,
    label=f"AP = {average_precision:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Indian Flood Impact Model"
)

plt.legend(
    loc="lower left"
)

plt.tight_layout()


PR_FILE = (
    FIGURES_DIR
    / "india_precision_recall_curve.png"
)

plt.savefig(
    PR_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Precision-recall curve saved to:"
)

print(
    PR_FILE
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if not IMPORTANCE_FILE.exists():

    print(
        "\nFeature importance file not found."
    )

    print(
        "Skipping feature importance visualization."
    )

else:

    feature_importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    if (
        "Feature" not in feature_importance.columns
        or
        "Importance" not in feature_importance.columns
    ):

        raise ValueError(
            "Feature importance file must contain "
            "'Feature' and 'Importance' columns."
        )


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nTop 20 features:"
    )

    print(
        feature_importance
        .head(20)
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # Save evaluation feature importance
    # --------------------------------------------------------

    EVALUATION_IMPORTANCE_FILE = (
        RESULTS_DIR
        / "india_evaluation_feature_importance.csv"
    )

    feature_importance.to_csv(
        EVALUATION_IMPORTANCE_FILE,
        index=False
    )


    print(
        "\nFeature importance table saved to:"
    )

    print(
        EVALUATION_IMPORTANCE_FILE
    )


    # --------------------------------------------------------
    # Plot top 15 features
    # --------------------------------------------------------

    top_features = (
        feature_importance
        .head(15)
        .sort_values(
            by="Importance",
            ascending=True
        )
    )


    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Feature Importances - Indian Flood Impact Model"
    )

    plt.tight_layout()


    IMPORTANCE_PLOT = (
        FIGURES_DIR
        / "india_feature_importance.png"
    )

    plt.savefig(
        IMPORTANCE_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nFeature importance plot saved to:"
    )

    print(
        IMPORTANCE_PLOT
    )


# ============================================================
# 21. LEAKAGE VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEAKAGE VERIFICATION")
print("=" * 70)

remaining_leakage = [
    column
    for column in LEAKAGE_COLUMNS
    if column in X_test.columns
]

if remaining_leakage:

    print(
        "WARNING: Leakage columns remain:"
    )

    for column in remaining_leakage:

        print(
            "-",
            column
        )

else:

    print(
        "PASS: No identified leakage columns "
        "are present in the evaluation features."
    )


# ============================================================
# 22. SAVE EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Average Precision"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            average_precision
        ]
    }
)


evaluation_summary.to_csv(
    EVALUATION_FILE,
    index=False
)


print(
    "\nEvaluation summary saved to:"
)

print(
    EVALUATION_FILE
)


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("INDIAN MODEL EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nTest samples:",
    len(X_test)
)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)

print("\nGenerated files:")

print(
    "-",
    CONFUSION_FILE
)

print(
    "-",
    ROC_FILE
)

print(
    "-",
    PR_FILE
)

if IMPORTANCE_FILE.exists():

    print(
        "-",
        IMPORTANCE_PLOT
    )

    print(
        "-",
        EVALUATION_IMPORTANCE_FILE
    )

print(
    "-",
    EVALUATION_FILE
)

print(
    "\nNext stage:"
)

print(
    "Robust time-based validation"
)

print("=" * 70)