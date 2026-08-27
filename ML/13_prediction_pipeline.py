from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# FINAL PREDICTION PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("FINAL PREDICTION PIPELINE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD FINAL MODEL
# ------------------------------------------------------------

model_path = MODEL_DIR / "final_flood_model.pkl"

model = joblib.load(model_path)

print("\n✅ Final model loaded:")
print(model_path)

# ------------------------------------------------------------
# 2. LOAD TEST DATA
# ------------------------------------------------------------

test_path = DATA_DIR / "test.csv"

test_df = pd.read_csv(test_path)

print("\nTest dataset loaded:")
print("Shape:", test_df.shape)

# ------------------------------------------------------------
# 3. LOAD PREPROCESSED TEST FEATURES
# ------------------------------------------------------------

X_test = np.load(
    MODEL_DIR / "X_test_processed.npy"
)

y_test = pd.read_csv(
    MODEL_DIR / "y_test.csv"
).squeeze()

print("\nProcessed test features:", X_test.shape)
print("Test labels:", len(y_test))

# ------------------------------------------------------------
# 4. GENERATE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully.")

# ------------------------------------------------------------
# 5. CREATE PREDICTION RESULTS
# ------------------------------------------------------------

results = test_df.copy()

results["Actual_Flood"] = y_test.values
results["Predicted_Flood"] = y_pred
results["Flood_Probability"] = y_probability

# ------------------------------------------------------------
# 6. RISK LEVEL
# ------------------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


results["Risk_Level"] = results[
    "Flood_Probability"
].apply(get_risk_level)

# ------------------------------------------------------------
# 7. DISPLAY SAMPLE PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

display_columns = [
    "Station_Names",
    "Year",
    "Month",
    "Rainfall",
    "Relative_Humidity",
    "Actual_Flood",
    "Predicted_Flood",
    "Flood_Probability",
    "Risk_Level"
]

print(
    results[display_columns]
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 8. SAVE PREDICTIONS
# ------------------------------------------------------------

prediction_path = MODEL_DIR / "test_predictions.csv"

results.to_csv(
    prediction_path,
    index=False
)

print("\n" + "=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(prediction_path)

# ------------------------------------------------------------
# 9. PREDICTION SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "\nPredicted flood cases:",
    int(y_pred.sum())
)

print(
    "Predicted non-flood cases:",
    int((y_pred == 0).sum())
)

print(
    "\nAverage flood probability:",
    round(y_probability.mean(), 4)
)

print("\nRisk distribution:")

print(
    results["Risk_Level"]
    .value_counts()
)

# ------------------------------------------------------------
# 10. VALIDATION
# ------------------------------------------------------------

correct_predictions = (
    y_pred == y_test.values
).sum()

total_predictions = len(y_test)

accuracy = (
    correct_predictions /
    total_predictions
)

print("\n" + "=" * 70)
print("PIPELINE VALIDATION")
print("=" * 70)

print(
    "Correct predictions:",
    correct_predictions
)

print(
    "Total predictions:",
    total_predictions
)

print(
    "Validation accuracy:",
    round(accuracy, 4)
)

print("\n" + "=" * 70)
print("FINAL PREDICTION PIPELINE COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("➡ Build real-time/user-input prediction system")

print("=" * 70)