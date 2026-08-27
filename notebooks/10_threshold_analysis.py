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
    average_precision_score,
    confusion_matrix
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
    / "india_flood_impact_final_time_based.pkl"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = (
    RESULTS_DIR
    / "india_threshold_analysis.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "High_Impact"

# Same chronological split used by the final model
TEST_RATIO = 0.20

# Probability thresholds to evaluate
THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - THRESHOLD ANALYSIS")
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
        f"Final model not found:\n{MODEL_FILE}"
    )

print("Feature dataset found.")
print("Final time-based model found.")


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
# 6. CHECK REQUIRED COLUMNS
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
# 7. CLEAN YEAR
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


# ============================================================
# 9. TIME-BASED SPLIT
# ============================================================

years = sorted(
    df["Year"].unique()
)

number_of_years = len(years)

test_year_count = max(
    1,
    int(
        np.ceil(
            number_of_years * TEST_RATIO
        )
    )
)

split_index = (
    number_of_years - test_year_count
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


train_df = df[
    df["Year"].isin(train_years)
].copy()

test_df = df[
    df["Year"].isin(test_years)
].copy()


print("\n" + "=" * 70)
print("TIME-BASED TEST SET")
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
# 10. PREPARE TEST FEATURES
# ============================================================

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET].astype(int)


# ============================================================
# 11. REMOVE NON-PREDICTIVE COLUMNS
# ============================================================

REMOVE_COLUMNS = [
    "Event Source",
    "Districts"
]

removed = []

for column in REMOVE_COLUMNS:

    if column in X_test.columns:

        X_test = X_test.drop(
            columns=[column]
        )

        removed.append(
            column
        )


print("\nRemoved non-predictive columns:")

if removed:

    for column in removed:

        print(
            "-",
            column
        )

else:

    print("None")


# ============================================================
# 12. LEAKAGE CONTROL
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

removed_leakage = []

for column in LEAKAGE_COLUMNS:

    if column in X_test.columns:

        X_test = X_test.drop(
            columns=[column]
        )

        removed_leakage.append(
            column
        )


print("\nLeakage control:")

if removed_leakage:

    print(
        "Removed leakage-related columns:"
    )

    for column in removed_leakage:

        print(
            "-",
            column
        )

else:

    print(
        "PASS: No direct casualty leakage columns found."
    )


# ============================================================
# 13. LOAD FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINAL MODEL")
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
# 14. GENERATE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PROBABILITY PREDICTIONS")
print("=" * 70)

try:

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

except Exception as error:

    print(
        "\nPrediction failed:"
    )

    print(error)

    raise


print(
    "\nProbability predictions generated successfully."
)


# ============================================================
# 15. OVERALL PROBABILITY METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

average_precision = (
    average_precision_score(
        y_test,
        probabilities
    )
)

print("\n" + "=" * 70)
print("PROBABILITY MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)


# ============================================================
# 16. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

results = []


for threshold in THRESHOLDS:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({

        "Threshold": round(
            float(threshold),
            2
        ),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "True_Negatives": tn,

        "False_Positives": fp,

        "False_Negatives": fn,

        "True_Positives": tp,

        "ROC_AUC": roc_auc,

        "Average_Precision": average_precision

    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

display_columns = [
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "False_Negatives",
    "True_Positives"
]

print(
    "\n"
)

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 18. FIND BEST THRESHOLD BY F1
# ============================================================

best_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

best_threshold = (
    best_row["Threshold"]
)

best_f1 = (
    best_row["F1_Score"]
)

best_precision = (
    best_row["Precision"]
)

best_recall = (
    best_row["Recall"]
)


# ============================================================
# 19. DEFAULT 0.50 THRESHOLD
# ============================================================

default_row = results_df[
    results_df["Threshold"] == 0.50
].iloc[0]


default_f1 = (
    default_row["F1_Score"]
)

default_precision = (
    default_row["Precision"]
)

default_recall = (
    default_row["Recall"]
)


# ============================================================
# 20. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nBest threshold by F1 : {best_threshold:.2f}"
)

print(
    f"F1 Score             : {best_f1:.4f}"
)

print(
    f"Precision            : {best_precision:.4f}"
)

print(
    f"Recall               : {best_recall:.4f}"
)


print("\n" + "=" * 70)
print("DEFAULT THRESHOLD (0.50)")
print("=" * 70)

print(
    f"\nF1 Score  : {default_f1:.4f}"
)

print(
    f"Precision : {default_precision:.4f}"
)

print(
    f"Recall    : {default_recall:.4f}"
)


print("\n" + "=" * 70)
print("THRESHOLD IMPROVEMENT")
print("=" * 70)

print(
    f"\nF1 improvement:"
    f" {best_f1 - default_f1:+.4f}"
)


# ============================================================
# 21. OPERATIONAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("OPERATIONAL INTERPRETATION")
print("=" * 70)

print(
    "\nLower thresholds generally increase recall "
    "while decreasing precision."
)

print(
    "Higher thresholds generally increase precision "
    "while decreasing recall."
)

print(
    "\nFor disaster-response applications, recall is "
    "particularly important because missing a genuinely "
    "high-impact event can be more serious than generating "
    "an additional warning."
)

print(
    "\nThe threshold should therefore be selected based "
    "on the operational objective rather than F1 score alone."
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nThreshold analysis saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nTesting period:"
    f" {TEST_START} - {TEST_END}"
)

print(
    f"Testing samples:"
    f" {len(test_df)}"
)

print(
    f"\nROC-AUC:"
    f" {roc_auc:.4f}"
)

print(
    f"Average Precision:"
    f" {average_precision:.4f}"
)

print(
    f"\nBest F1 threshold:"
    f" {best_threshold:.2f}"
)

print(
    f"Best F1:"
    f" {best_f1:.4f}"
)

print(
    "\nNo threshold is automatically declared "
    "the final operational threshold."
)

print(
    "The analysis provides evidence for selecting "
    "an appropriate threshold based on the project's "
    "disaster-response objective."
)

print("=" * 70)