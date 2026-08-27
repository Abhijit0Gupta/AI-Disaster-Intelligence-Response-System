from pathlib import Path
import pandas as pd

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# DATA PREPROCESSING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FloodPrediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("DATA PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. KEEP ONLY LABELED RECORDS
# ------------------------------------------------------------

df = df[df["Flood?"].notna()].copy()

print("Labeled dataset shape:", df.shape)

# ------------------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates().copy()

print("Shape after duplicate removal:", df.shape)

# ------------------------------------------------------------
# 4. CONVERT TARGET TO INTEGER
# ------------------------------------------------------------

df["Flood?"] = df["Flood?"].astype(int)

# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values after selecting labeled data:")

missing = df.isnull().sum()

print(missing[missing > 0])

# ------------------------------------------------------------
# 6. CHECK TARGET
# ------------------------------------------------------------

print("\nTarget distribution:")

print(df["Flood?"].value_counts())

# ------------------------------------------------------------
# 7. SAVE CLEAN DATASET
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "flood_labeled_clean.csv"

df.to_csv(output_path, index=False)

print("\nClean dataset saved to:")
print(output_path)

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Output file:", output_path)

print("\nNext stage:")
print("➡ Feature engineering")

print("=" * 70)