from pathlib import Path

import pandas as pd
import numpy as np

from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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

OUTPUT_FILE = (
    RESULTS_DIR
    / "india_robust_validation_summary.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "High_Impact"

# Use the latest 20% of years for testing.
# The exact cutoff is calculated from the dataset.

TEST_RATIO = 0.20


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - ROBUST TIME-BASED VALIDATION")
print("=" * 70)


# ============================================================
# 4. CHECK FILES
# ============================================================

print("\nChecking required files...")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

print("Feature dataset found.")
print("Model found.")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    FEATURE_FILE
)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# 6. CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

if "Year" not in df.columns:

    raise ValueError(
        "Year column is required for time-based validation."
    )


# ============================================================
# 7. CLEAN YEAR FOR SPLITTING
# ============================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

missing_year = df["Year"].isna().sum()

if missing_year > 0:

    print(
        f"\nRemoving {missing_year} rows with missing Year."
    )

    df = df.dropna(
        subset=["Year"]
    ).reset_index(
        drop=True
    )

df["Year"] = df["Year"].astype(int)


# ============================================================
# 8. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    by="Year"
).reset_index(
    drop=True
)


print("\nYear range:")

print(
    "Minimum year:",
    df["Year"].min()
)

print(
    "Maximum year:",
    df["Year"].max()
)


# ============================================================
# 9. DETERMINE TIME-BASED SPLIT
# ============================================================

years = sorted(
    df["Year"].unique()
)

number_of_years = len(years)

test_year_count = max(
    1,
    int(
        np.ceil(
            number_of_years
            * TEST_RATIO
        )
    )
)

split_index = (
    number_of_years
    - test_year_count
)

train_years = years[
    :split_index
]

test_years = years[
    split_index:
]


TRAIN_START = train_years[0]
TRAIN_END = train_years[-1]

TEST_START = test_years[0]
TEST_END = test_years[-1]


# ============================================================
# 10. CREATE TRAIN / TEST DATA
# ============================================================

train_df = df[
    df["Year"].isin(train_years)
].copy()

test_df = df[
    df["Year"].isin(test_years)
].copy()


print("\n" + "=" * 70)
print("TIME-BASED SPLIT")
print("=" * 70)

print(
    f"\nTraining years : {TRAIN_START} - {TRAIN_END}"
)

print(
    f"Testing years  : {TEST_START} - {TEST_END}"
)

print(
    "\nTraining rows :",
    len(train_df)
)

print(
    "Testing rows  :",
    len(test_df)
)


# ============================================================
# 11. TARGET DISTRIBUTION
# ============================================================

print("\nTraining target distribution:")

print(
    train_df[TARGET]
    .value_counts()
    .sort_index()
)


print("\nTesting target distribution:")

print(
    test_df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. PREPARE FEATURES
# ============================================================

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET].astype(int)


# ============================================================
# 13. LEAKAGE CONTROL
# ============================================================

LEAKAGE_COLUMNS = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Total_Human_Impact",
    "Reported_Human_Impact"
]


removed_train = []
removed_test = []


for column in LEAKAGE_COLUMNS:

    if column in X_train.columns:

        X_train = X_train.drop(
            columns=[column]
        )

        removed_train.append(
            column
        )

    if column in X_test.columns:

        X_test = X_test.drop(
            columns=[column]
        )

        removed_test.append(
            column
        )


print("\n" + "=" * 70)
print("LEAKAGE CONTROL")
print("=" * 70)


if removed_train:

    print(
        "\nExcluded leakage-related variables:"
    )

    for column in removed_train:

        print(
            "-",
            column
        )

else:

    print(
        "\nNo casualty leakage variables found."
    )


# ============================================================
# 14. LOAD TRAINED PIPELINE
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
# 15. CHECK FEATURE COMPATIBILITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COMPATIBILITY CHECK")
print("=" * 70)

print(
    "\nTraining features:",
    len(X_train.columns)
)

print(
    "Testing features :",
    len(X_test.columns)
)


# ============================================================
# 16. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TIME-BASED PREDICTIONS")
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
        "\nPrediction failed:"
    )

    print(error)

    raise


print(
    "\nPredictions generated successfully!"
)


# ============================================================
# 17. CALCULATE METRICS
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
# 18. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ROBUST TIME-BASED VALIDATION RESULTS")
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
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 20. CLASSIFICATION REPORT
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
# 21. COMPARE TRAINING AND TEST YEARS
# ============================================================

print("\n" + "=" * 70)
print("TIME SPLIT SUMMARY")
print("=" * 70)

print(
    f"\nTraining period:"
    f" {TRAIN_START} - {TRAIN_END}"
)

print(
    f"Testing period:"
    f" {TEST_START} - {TEST_END}"
)

print(
    "\nThe model was evaluated on later years "
    "that were not part of the training period."
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Training_Start_Year",
        "Training_End_Year",
        "Testing_Start_Year",
        "Testing_End_Year",
        "Training_Rows",
        "Testing_Rows",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "Average_Precision"

    ],

    "Value": [

        TRAIN_START,
        TRAIN_END,
        TEST_START,
        TEST_END,
        len(train_df),
        len(test_df),
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        average_precision

    ]

})


summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 23. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("ROBUST VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nThis validation uses a chronological split."
)

print(
    "Earlier years were used for training, "
    "while later years were reserved for testing."
)

print(
    "\nThis provides a more realistic estimate "
    "of how the model may perform on future flood records."
)

print(
    "\nFINAL TIME-BASED RESULTS:"
)

print(
    f"Accuracy          : {accuracy:.4f}"
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

print("=" * 70)