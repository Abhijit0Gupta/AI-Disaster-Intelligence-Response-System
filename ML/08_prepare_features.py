from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# FEATURE PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("FEATURE PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining shape:", train_df.shape)
print("Testing shape :", test_df.shape)

# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

TARGET = "Flood?"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ------------------------------------------------------------
# 3. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "Station_Names",
    "Season",
    "Rainfall_Category"
]

numerical_features = [
    "Year",
    "Month",
    "Max_Temp",
    "Min_Temp",
    "Rainfall",
    "Relative_Humidity",
    "Wind_Speed",
    "Cloud_Coverage",
    "Bright_Sunshine",
    "LATITUDE",
    "LONGITUDE",
    "ALT",
    "Temperature_Range",
    "Humidity_Rainfall_Index"
]

print("\nCategorical features:")
for feature in categorical_features:
    print("-", feature)

print("\nNumerical features:")
for feature in numerical_features:
    print("-", feature)

# ------------------------------------------------------------
# 4. CREATE PREPROCESSING PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# ------------------------------------------------------------
# 5. FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

# IMPORTANT:
# Test data is ONLY transformed.
# It is never used to fit the preprocessing pipeline.

X_test_processed = preprocessor.transform(X_test)

print("✅ Preprocessing complete.")

# ------------------------------------------------------------
# 6. DISPLAY PROCESSED SHAPES
# ------------------------------------------------------------

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)

# ------------------------------------------------------------
# 7. SAVE PREPROCESSOR
# ------------------------------------------------------------

preprocessor_path = MODEL_DIR / "preprocessor.pkl"

joblib.dump(preprocessor, preprocessor_path)

print("\nPreprocessor saved to:")
print(preprocessor_path)

# ------------------------------------------------------------
# 8. SAVE PROCESSED ARRAYS
# ------------------------------------------------------------

from scipy.sparse import save_npz

save_npz(
    MODEL_DIR / "X_train_processed.npz",
    X_train_processed
)

save_npz(
    MODEL_DIR / "X_test_processed.npz",
    X_test_processed
)

y_train.to_csv(
    MODEL_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    MODEL_DIR / "y_test.csv",
    index=False
)

print("\nProcessed datasets saved.")

# ------------------------------------------------------------
# 9. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE PREPARATION COMPLETE")
print("=" * 70)

print("\nTraining samples:", X_train_processed.shape[0])
print("Testing samples :", X_test_processed.shape[0])
print("Processed features:", X_train_processed.shape[1])

print("\nNext stage:")
print("➡ Train baseline ML models")

print("=" * 70)