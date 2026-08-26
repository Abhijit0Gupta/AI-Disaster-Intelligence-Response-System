from pathlib import Path
import pandas as pd

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# TRAIN-TEST SPLIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("TRAIN-TEST SPLIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD FEATURE-ENGINEERED DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# 2. SORT CHRONOLOGICALLY
# ------------------------------------------------------------

df = df.sort_values(
    by=["Year", "Month", "Station_Names"]
).reset_index(drop=True)

print("\nData sorted chronologically.")

print(
    "Time range:",
    f"{df['Year'].min()}-{df['Month'].min():02d}",
    "to",
    f"{df['Year'].max()}-{df['Month'].max():02d}"
)

# ------------------------------------------------------------
# 3. DEFINE TRAIN-TEST YEARS
# ------------------------------------------------------------

TRAIN_END_YEAR = 2008

train_df = df[df["Year"] <= TRAIN_END_YEAR].copy()
test_df = df[df["Year"] > TRAIN_END_YEAR].copy()

# ------------------------------------------------------------
# 4. DISPLAY SPLIT INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT INFORMATION")
print("=" * 60)

print("\nTraining data:")
print("Rows:", len(train_df))
print(
    "Years:",
    train_df["Year"].min(),
    "to",
    train_df["Year"].max()
)

print("\nTesting data:")
print("Rows:", len(test_df))
print(
    "Years:",
    test_df["Year"].min(),
    "to",
    test_df["Year"].max()
)

# ------------------------------------------------------------
# 5. TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 60)

print(train_df["Flood?"].value_counts())
print("\nPercentage:")
print(train_df["Flood?"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("TESTING TARGET DISTRIBUTION")
print("=" * 60)

print(test_df["Flood?"].value_counts())
print("\nPercentage:")
print(test_df["Flood?"].value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 6. CHECK FOR TIME LEAKAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TIME LEAKAGE CHECK")
print("=" * 60)

print("Latest training year:", train_df["Year"].max())
print("Earliest testing year:", test_df["Year"].min())

if train_df["Year"].max() < test_df["Year"].min():
    print("✅ No chronological overlap detected.")
else:
    print("❌ WARNING: Chronological overlap detected!")

# ------------------------------------------------------------
# 7. SAVE TRAIN AND TEST DATA
# ------------------------------------------------------------

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("\nTraining dataset:")
print(train_path)

print("\nTesting dataset:")
print(test_path)

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT COMPLETE")
print("=" * 70)

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nNext stage:")
print("➡ Feature encoding and model preparation")

print("=" * 70)