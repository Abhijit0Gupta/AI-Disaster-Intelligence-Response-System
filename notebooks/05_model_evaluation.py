from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import load

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

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EVALUATION_FILE = (
    RESULTS_DIR
    / "india_evaluation_summary.csv"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "models"
    / "india_feature_importance.csv"
)


# Create output directories

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - MODEL EVALUATION")
print("=" * 70)


# ============================================================
# 3. CHECK FILES
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
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(FEATURE_FILE)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# 5. TARGET
# ============================================================

TARGET = "High_Impact"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print("\nTarget column:", TARGET)

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = load(MODEL_FILE)

print("\nModel loaded successfully!")

print("Model type:")

print(type(model).__name__)


# ============================================================
# 7. PREPARE X AND y
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


print("\n" + "=" * 70)
print("FEATURE / TARGET INFORMATION")
print("=" * 70)

print("\nTotal samples :", len(X))

print("Total features:", len(X.columns))


# ============================================================
# 8. REMOVE TARGET / LEAKAGE COLUMNS
# ============================================================

# These variables directly contain casualty information
# and must not be used to predict High_Impact because
# High_Impact itself was created from Human fatality.

LEAKAGE_COLUMNS = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Total_Human_Impact",
    "Reported_Human_Impact"
]


removed_columns = []

for column in LEAKAGE_COLUMNS:

    if column in X.columns:

        X = X.drop(
            columns=[column]
        )

        removed_columns.append(column)


if removed_columns:

    print("\nLeakage-related columns excluded:")

    for column in removed_columns:

        print("-", column)

else:

    print("\nNo leakage-related columns found.")


# ============================================================
# 9. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATA CHECK")
print("=" * 70)

print("\nRemaining features:")

print(
    X.columns.tolist()
)

print("\nMissing values:")

missing_values = X.isnull().sum()

missing_values = (
    missing_values[
        missing_values > 0
    ]
)

if missing_values.empty:

    print("No missing values.")

else:

    print(
        missing_values.to_string()
    )


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

try:

    y_pred = model.predict(X)

    y_probability = (
        model.predict_proba(X)[:, 1]
    )

except Exception as error:

    print("\nERROR while generating predictions.")

    print(error)

    print(
        "\nThis usually means the evaluation data "
        "does not match the features used during training."
    )

    raise


print("\nPredictions generated successfully!")


# ============================================================
# 11. PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    y_probability
)

average_precision = (
    average_precision_score(
        y,
        y_probability
    )
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


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y,
    y_pred
)

print("\n")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX PLOT
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
    ["Predicted 0", "Predicted 1"]
)

plt.yticks(
    [0, 1],
    ["Actual 0", "Actual 1"]
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

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
# 14. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y,
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
# 15. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y,
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
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


feature_importance = None


# ------------------------------------------------------------
# Option 1: Read saved feature importance
# ------------------------------------------------------------

if IMPORTANCE_FILE.exists():

    try:

        saved_importance = pd.read_csv(
            IMPORTANCE_FILE
        )

        if (
            "Feature" in saved_importance.columns
            and
            "Importance" in saved_importance.columns
        ):

            feature_importance = (
                saved_importance
                .copy()
            )

            print(
                "\nLoaded feature importance from:"
            )

            print(
                IMPORTANCE_FILE
            )

    except Exception:

        feature_importance = None


# ------------------------------------------------------------
# Option 2: Extract from model
# ------------------------------------------------------------

if feature_importance is None:

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance_values = (
            model.feature_importances_
        )

        # Try to obtain feature names from
        # preprocessing pipeline.

        feature_names = None

        try:

            if hasattr(
                model,
                "named_steps"
            ):

                preprocessor = (
                    model.named_steps
                    .get("preprocessor")
                )

                classifier = (
                    model.named_steps
                    .get("classifier")
                )

                if preprocessor is not None:

                    feature_names = (
                        preprocessor
                        .get_feature_names_out()
                    )

                    importance_values = (
                        classifier
                        .feature_importances_
                    )

        except Exception:

            feature_names = None


        if feature_names is None:

            feature_names = np.array(
                X.columns
            )


        if len(feature_names) != len(
            importance_values
        ):

            raise ValueError(
                "Feature importance length does not "
                "match feature names."
            )


        feature_importance = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importance_values

        })


# ------------------------------------------------------------
# Sort importance
# ------------------------------------------------------------

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


print("\nTop 20 features:")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 17. SAVE FEATURE IMPORTANCE
# ============================================================

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


# ============================================================
# 18. FEATURE IMPORTANCE PLOT
# ============================================================

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
# 19. LEAKAGE INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE INVESTIGATION")
print("=" * 70)


print(
    "\nTarget definition:"
)

print(
    "High_Impact = 1 when Human fatality >= 10"
)

print(
    "High_Impact = 0 when Human fatality < 10"
)


print(
    "\nThe following casualty variables must NOT "
    "be model inputs:"
)

for column in LEAKAGE_COLUMNS:

    if column in df.columns:

        print(
            "-",
            column
        )


print(
    "\nThese variables were excluded from evaluation."
)


# ============================================================
# 20. CHECK IMPORTANT FEATURES
# ============================================================

print(
    "\nTop 10 features requiring interpretation:"
)

print(
    feature_importance
    .head(10)
    .to_string(index=False)
)


print(
    "\nIMPORTANT:"
)

print(
    "High feature importance does not automatically "
    "mean data leakage."
)

print(
    "However, features that are proxies for the target "
    "definition should be investigated before final reporting."
)


# ============================================================
# 21. SAVE EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame({

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

})


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
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("INDIAN MODEL EVALUATION COMPLETE")
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


print("\nNext stage:")

print(
    "Robust time-based validation"
)

print("=" * 70)