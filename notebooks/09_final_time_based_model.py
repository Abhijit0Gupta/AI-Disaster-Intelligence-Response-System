from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "India_Flood_Inventory_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_FILE = (
    MODEL_DIR
    / "india_flood_impact_final_time_based.pkl"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "india_final_time_based_summary.csv"
)

IMPORTANCE_FILE = (
    RESULTS_DIR
    / "india_final_time_based_feature_importance.csv"
)

CONFUSION_FILE = (
    FIGURES_DIR
    / "india_final_confusion_matrix.png"
)

ROC_FILE = (
    FIGURES_DIR
    / "india_final_roc_curve.png"
)

PR_FILE = (
    FIGURES_DIR
    / "india_final_precision_recall_curve.png"
)

IMPORTANCE_PLOT = (
    FIGURES_DIR
    / "india_final_feature_importance.png"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "High_Impact"

TEST_RATIO = 0.20

RANDOM_STATE = 42


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - FINAL TIME-BASED MODEL")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This script trains the model ONLY on earlier years"
)

print(
    "and evaluates it ONLY on later unseen years."
)


# ============================================================
# 4. CHECK INPUT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILE")
print("=" * 70)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature dataset not found:\n{INPUT_FILE}"
    )

print(
    "\nFeature dataset found."
)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\nDataset loaded successfully!"
)

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)


# ============================================================
# 6. TARGET CHECK
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

if "Year" not in df.columns:

    raise ValueError(
        "Year column is required for time-based validation."
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
# 7. CLEAN YEAR
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TIME INFORMATION")
print("=" * 70)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

missing_year = df["Year"].isna().sum()

if missing_year > 0:

    print(
        f"\nRemoving {missing_year} rows "
        "with missing Year."
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

years = sorted(
    df["Year"].unique()
)

print(
    "\nMinimum year:",
    df["Year"].min()
)

print(
    "Maximum year:",
    df["Year"].max()
)

print(
    "Number of unique years:",
    len(years)
)


# ============================================================
# 9. DETERMINE TIME SPLIT
# ============================================================

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
# 10. CREATE TIME-BASED DATASETS
# ============================================================

train_df = df[
    df["Year"].isin(train_years)
].copy()

test_df = df[
    df["Year"].isin(test_years)
].copy()


print("\n" + "=" * 70)
print("FINAL TIME-BASED SPLIT")
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

print(
    "\nTraining target distribution:"
)

print(
    train_df[TARGET]
    .value_counts()
    .sort_index()
)

print(
    "\nTesting target distribution:"
)

print(
    test_df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. PREPARE X AND y
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
# 13. REMOVE NON-PREDICTIVE / HIGH-CARDINALITY COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("REMOVING NON-PREDICTIVE COLUMNS")
print("=" * 70)

COLUMNS_TO_DROP = [
    "Event Source",
    "Districts"
]

removed_columns = []

for column in COLUMNS_TO_DROP:

    if column in X_train.columns:

        X_train = X_train.drop(
            columns=[column]
        )

        removed_columns.append(
            column
        )

    if column in X_test.columns:

        X_test = X_test.drop(
            columns=[column]
        )


if removed_columns:

    for column in removed_columns:

        print(
            "Removed:",
            column
        )

else:

    print(
        "No specified high-cardinality columns found."
    )


# ============================================================
# 14. LEAKAGE CONTROL
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CONTROL")
print("=" * 70)

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

    if column in X_train.columns:

        X_train = X_train.drop(
            columns=[column]
        )

        removed_leakage.append(
            column
        )

    if column in X_test.columns:

        X_test = X_test.drop(
            columns=[column]
        )


if removed_leakage:

    print(
        "\nExcluded leakage-related columns:"
    )

    for column in removed_leakage:

        print(
            "-",
            column
        )

else:

    print(
        "\nPASS: No direct casualty leakage columns found."
    )


# ============================================================
# 15. FEATURE COMPATIBILITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COMPATIBILITY")
print("=" * 70)

print(
    "\nTraining features:",
    len(X_train.columns)
)

print(
    "Testing features :",
    len(X_test.columns)
)

if list(X_train.columns) != list(X_test.columns):

    raise ValueError(
        "Training and testing feature columns do not match."
    )

print(
    "\nPASS: Training and testing features match."
)


# ============================================================
# 16. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\nNumeric features:")

print(
    numeric_features
)

print("\nCategorical features:")

print(
    categorical_features
)


# ============================================================
# 17. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 70)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 18. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 19. TRAIN FINAL TIME-BASED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL TIME-BASED MODEL")
print("=" * 70)

print(
    f"\nTraining ONLY on years "
    f"{TRAIN_START}-{TRAIN_END}"
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "\nTraining model..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# ============================================================
# 20. PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON UNSEEN FUTURE YEARS")
print("=" * 70)

print(
    f"\nTesting ONLY on years "
    f"{TEST_START}-{TEST_END}"
)

print(
    "\nTesting samples:",
    len(X_test)
)

y_pred = pipeline.predict(
    X_test
)

y_probability = (
    pipeline.predict_proba(
        X_test
    )[:, 1]
)

print(
    "\nPredictions generated successfully."
)


# ============================================================
# 21. PERFORMANCE METRICS
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


print("\n" + "=" * 70)
print("FINAL TIME-BASED MODEL PERFORMANCE")
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
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(cm)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 23. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Final Time-Based Model"
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
# 24. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
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
    "ROC Curve - Final Time-Based Model"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

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
# 25. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = (
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
    "Precision-Recall Curve - Final Time-Based Model"
)

plt.legend(
    loc="lower left"
)

plt.tight_layout()

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
# 26. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)

rf_model = (
    pipeline.named_steps[
        "model"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_values = (
    rf_model.feature_importances_
)

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
    }
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
    "\nTop 20 encoded features:"
)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 27. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)

print(
    "\nFeature importance saved to:"
)

print(
    IMPORTANCE_FILE
)


# ============================================================
# 28. FEATURE IMPORTANCE PLOT
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
    "Top 15 Feature Importances - Final Time-Based Model"
)

plt.tight_layout()

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
# 29. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    "\nFinal time-based model saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# 30. SAVE PERFORMANCE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "Metric": [
            "Training_Start_Year",
            "Training_End_Year",
            "Testing_Start_Year",
            "Testing_End_Year",
            "Training_Rows",
            "Testing_Rows",
            "Features_Before_Encoding",
            "Encoded_Features",
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
            len(X_train),
            len(X_test),
            len(X_train.columns),
            len(feature_names),
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            average_precision
        ]
    }
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(
    "\nPerformance summary saved to:"
)

print(
    SUMMARY_FILE
)


# ============================================================
# 31. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL VERIFICATION")
print("=" * 70)

print(
    "\nTraining period:"
)

print(
    f"{TRAIN_START} - {TRAIN_END}"
)

print(
    "\nTesting period:"
)

print(
    f"{TEST_START} - {TEST_END}"
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
    "\nThe model was FIT only on the training period."
)

print(
    "The test period contains later years not used for fitting."
)

print(
    "\nFinal performance:"
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


# ============================================================
# 32. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TIME-BASED MODEL COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    "-",
    MODEL_FILE
)

print(
    "-",
    SUMMARY_FILE
)

print(
    "-",
    IMPORTANCE_FILE
)

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
    "\nThis model is the proper candidate for final evaluation"
)

print(
    "because the training and testing periods are strictly separated."
)

print("=" * 70)