from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = (
    RESULTS_DIR
    / "india_model_sensitivity_summary.csv"
)

RESULTS_DIR.mkdir(
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
print("INDIA FLOOD IMPACT - MODEL SENSITIVITY ANALYSIS")
print("=" * 70)


# ============================================================
# 4. CHECK INPUT
# ============================================================

print("\nChecking input file...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

print("Feature dataset found.")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
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
# 6. TARGET CHECK
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

if "Year" not in df.columns:

    raise ValueError(
        "Year column is required for sensitivity analysis."
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

train_years = years[:split_index]

test_years = years[split_index:]


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
    f"\nTraining years : {train_years[0]} - {train_years[-1]}"
)

print(
    f"Testing years  : {test_years[0]} - {test_years[-1]}"
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
# 9. BASE FEATURES
# ============================================================

X_full = df.drop(
    columns=[TARGET]
)

y_full = df[TARGET].astype(int)


# ============================================================
# 10. COLUMNS THAT SHOULD NEVER BE USED
# ============================================================

ALWAYS_DROP = [
    "Event Source",
    "Districts",

    # Target/outcome information
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Extent of damage",
    "Total_Human_Impact",
    "Reported_Human_Impact",

    # Identifiers
    "Unnamed: 0",
    "UEI"
]


# ============================================================
# 11. SENSITIVITY EXPERIMENTS
# ============================================================

experiments = {

    "Baseline_All_Features": [],

    "Without_Date_Features": [
        "Start_Year",
        "Start_Month",
        "Start_Day",
        "Year",
        "Month",
        "Day",
        "Quarter",
        "Month_Sin",
        "Month_Cos"
    ],

    "Without_Geographical_Codes": [
        "State_Codes",
        "District_LGD_Codes"
    ],

    "Without_Duration": [
        "Duration(Days)",
        "Long_Event"
    ],

    "Without_Date_And_Geography": [
        "Start_Year",
        "Start_Month",
        "Start_Day",
        "Year",
        "Month",
        "Day",
        "Quarter",
        "Month_Sin",
        "Month_Cos",
        "State_Codes",
        "District_LGD_Codes"
    ],

    "Core_Event_Features": [
        "Start_Year",
        "Start_Month",
        "Year",
        "Month",
        "Duration(Days)",
        "Long_Event",
        "Main Cause",
        "State",
        "Cause_Group",
        "State_Count",
        "District_Count",
        "Event_Source_Available"
    ]
}


# ============================================================
# 12. RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, additional_drops in experiments.items():

    print("\n")
    print("=" * 70)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare train/test data
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET]
    ).copy()

    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(
        columns=[TARGET]
    ).copy()

    y_test = test_df[TARGET].astype(int)


    # --------------------------------------------------------
    # Remove always-excluded columns
    # --------------------------------------------------------

    columns_to_drop = (
        ALWAYS_DROP
        + additional_drops
    )

    columns_to_drop = list(
        dict.fromkeys(
            columns_to_drop
        )
    )

    for column in columns_to_drop:

        if column in X_train.columns:

            X_train = X_train.drop(
                columns=[column]
            )

        if column in X_test.columns:

            X_test = X_test.drop(
                columns=[column]
            )


    print(
        "\nFeatures used:",
        len(X_train.columns)
    )

    print(
        X_train.columns.tolist()
    )


    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    numeric_features = (
        X_train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(
            include=["object", "string"]
        )
        .columns
        .tolist()
    )


    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Complete pipeline
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults:")

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


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({

        "Experiment":
            experiment_name,

        "Features_Used":
            len(X_train.columns),

        "Training_Rows":
            len(X_train),

        "Testing_Rows":
            len(X_test),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "ROC_AUC":
            roc_auc,

        "Average_Precision":
            average_precision

    })


# ============================================================
# 13. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        by="F1_Score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 70)
print("MODEL SENSITIVITY COMPARISON")
print("=" * 70)

print(
    "\n",
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSensitivity analysis saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 15. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)

best_model = (
    results_df.iloc[0]
)

baseline = (
    results_df[
        results_df["Experiment"]
        == "Baseline_All_Features"
    ]
)

print(
    "\nBest experiment:",
    best_model["Experiment"]
)

print(
    f"Best F1 Score: "
    f"{best_model['F1_Score']:.4f}"
)

if not baseline.empty:

    baseline_f1 = (
        baseline.iloc[0]["F1_Score"]
    )

    print(
        f"\nBaseline F1 Score: "
        f"{baseline_f1:.4f}"
    )

    print(
        f"Difference: "
        f"{best_model['F1_Score'] - baseline_f1:+.4f}"
    )


print(
    "\nThe purpose of this analysis is to determine "
    "whether model performance depends strongly on "
    "specific groups of features."
)

print(
    "\nDo NOT select the final model solely because "
    "it has the highest score."
)

print(
    "The feature groups must also be checked for "
    "possible proxy leakage and real-world validity."
)

print("=" * 70)