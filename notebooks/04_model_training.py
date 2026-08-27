from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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
    confusion_matrix,
    classification_report
)

import joblib


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

MODEL_FILE = (
    MODEL_DIR
    / "india_flood_impact_random_forest.pkl"
)

FEATURE_FILE = (
    MODEL_DIR
    / "india_feature_importance.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_FILE = (
    RESULTS_DIR
    / "india_model_training_summary.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - MODEL TRAINING")
print("=" * 70)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "High_Impact"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print("\n" + "=" * 70)
print("TARGET INFORMATION")
print("=" * 70)

print("Target column:", TARGET)

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)

print("\n" + "=" * 70)
print("FEATURE / TARGET INFORMATION")
print("=" * 70)

print(
    "Number of features :",
    X.shape[1]
)

print(
    "Number of samples  :",
    X.shape[0]
)


# ============================================================
# 5. REMOVE NON-PREDICTIVE / HIGH-CARDINALITY COLUMNS
# ============================================================

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
            f"Removed column: {column}"
        )


# ============================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = (
    X
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

categorical_features = (
    X
    .select_dtypes(
        include=["object", "string"]
    )
    .columns
    .tolist()
)

print("\nNumeric features:")

for column in numeric_features:
    print("-", column)

print("\nCategorical features:")

for column in categorical_features:
    print("-", column)


# ============================================================
# 7. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
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


print("Numeric missing values -> median imputation")
print("Categorical missing values -> most-frequent imputation")
print("Categorical variables -> One-Hot Encoding")


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training samples :",
    len(X_train)
)

print(
    "Testing samples  :",
    len(X_test)
)

print("\nTraining target distribution:")

print(
    y_train
    .value_counts()
    .sort_index()
)

print("\nTesting target distribution:")

print(
    y_test
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
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


print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. PERFORMANCE
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


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 13. CLASSIFICATION REPORT
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
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

rf_model = pipeline.named_steps[
    "model"
]

fitted_preprocessor = pipeline.named_steps[
    "preprocessor"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importances = (
    rf_model.feature_importances_
)

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTop 20 features:")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving trained model...")

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    "Model saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# 16. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    FEATURE_FILE,
    index=False
)

print(
    "\nFeature importance saved to:"
)

print(
    FEATURE_FILE
)


# ============================================================
# 17. SAVE PERFORMANCE SUMMARY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

summary = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
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
# 18. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "Training samples :",
    len(X_train)
)

print(
    "Testing samples  :",
    len(X_test)
)

print(
    "Model features   :",
    X.shape[1]
)

print("\nPerformance:")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nSaved files:")

print(MODEL_FILE)
print(FEATURE_FILE)
print(SUMMARY_FILE)

print("\nNext stage:")
print("Model Evaluation / Visualization")