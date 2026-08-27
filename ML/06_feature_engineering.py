from pathlib import Path
import pandas as pd

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# FEATURE ENGINEERING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_labeled_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD CLEAN DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nInput shape:", df.shape)

# ------------------------------------------------------------
# 2. CREATE SEASON FEATURE
# ------------------------------------------------------------

def get_season(month):
    if month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Post-Monsoon"
    else:
        return "Winter"

df["Season"] = df["Month"].apply(get_season)

print("\nSeason distribution:")
print(df["Season"].value_counts())

# ------------------------------------------------------------
# 3. CREATE RAINFALL INTENSITY FEATURE
# ------------------------------------------------------------

def rainfall_category(rainfall):
    if rainfall <= 50:
        return "Low"
    elif rainfall <= 200:
        return "Moderate"
    elif rainfall <= 500:
        return "High"
    else:
        return "Extreme"

df["Rainfall_Category"] = df["Rainfall"].apply(rainfall_category)

print("\nRainfall category distribution:")
print(df["Rainfall_Category"].value_counts())

# ------------------------------------------------------------
# 4. CREATE TEMPERATURE RANGE
# ------------------------------------------------------------

df["Temperature_Range"] = df["Max_Temp"] - df["Min_Temp"]

# ------------------------------------------------------------
# 5. CREATE HUMIDITY-RISK FEATURE
# ------------------------------------------------------------

df["Humidity_Rainfall_Index"] = (
    df["Relative_Humidity"] * df["Rainfall"]
)

# ------------------------------------------------------------
# 6. REMOVE IDENTIFIER / REDUNDANT FEATURES
# ------------------------------------------------------------

columns_to_remove = [
    "Sl",
    "Period",
    "Station_Number",
    "X_COR",
    "Y_COR"
]

df = df.drop(columns=columns_to_remove)

# ------------------------------------------------------------
# 7. DISPLAY FINAL FEATURES
# ------------------------------------------------------------

print("\nFinal features:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# ------------------------------------------------------------
# 8. SAVE FEATURE-ENGINEERED DATASET
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "flood_features.csv"

df.to_csv(output_path, index=False)

print("\nFeature-engineered dataset saved to:")
print(output_path)

# ------------------------------------------------------------
# 9. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("Final shape:", df.shape)

print("\nNext stage:")
print("➡ Train-test split and ML model preparation")

print("=" * 70)