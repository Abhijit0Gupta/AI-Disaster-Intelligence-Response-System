from pathlib import Path
import pandas as pd

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# Dataset Analysis
# ============================================================

print("=" * 70)
print("       AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("                    DATASET ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. FIND PROJECT ROOT AND DATASET
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FloodPrediction.csv"

print("\nDataset path:")
print(DATA_PATH)

# Check whether file exists
if not DATA_PATH.exists():
    print("\n❌ ERROR: Dataset file was not found!")
    print("Expected location:")
    print(DATA_PATH)
    print("\nPlease make sure FloodPrediction.csv is inside:")
    print("data/raw/")
    raise SystemExit

print("\n✅ Dataset file found!")


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print("\n❌ ERROR while reading dataset:")
    print(e)
    raise SystemExit

print("✅ Dataset loaded successfully!")


# ------------------------------------------------------------
# 3. DATASET SIZE
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ------------------------------------------------------------
# 4. COLUMN NAMES
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("COLUMNS")
print("=" * 50)

for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")


# ------------------------------------------------------------
# 5. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)

print(df.head().to_string())


# ------------------------------------------------------------
# 6. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


# ------------------------------------------------------------
# 7. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# 8. DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("DUPLICATES")
print("=" * 50)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 9. FLOOD LABEL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("FLOOD LABEL ANALYSIS")
print("=" * 50)

if "Flood?" in df.columns:

    print("\nFlood label counts:")
    print(df["Flood?"].value_counts(dropna=False))

    print("\nUnique Flood values:")
    print(df["Flood?"].unique())

    print("\nMissing Flood labels:")
    print(df["Flood?"].isna().sum())

else:

    print("❌ WARNING: 'Flood?' column was not found.")


# ------------------------------------------------------------
# 10. NUMERICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("NUMERICAL SUMMARY")
print("=" * 50)

print(df.describe().to_string())


# ------------------------------------------------------------
# 11. UNIQUE VALUES
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("UNIQUE VALUES PER COLUMN")
print("=" * 50)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# ------------------------------------------------------------
# 12. RANDOM SAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("RANDOM SAMPLE")
print("=" * 50)

sample_size = min(5, len(df))

print(df.sample(sample_size, random_state=42).to_string())


# ------------------------------------------------------------
# 13. MEMORY USAGE
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("MEMORY USAGE")
print("=" * 50)

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"{memory_mb:.2f} MB")


# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("                    ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nDataset size: {df.shape[0]} rows × {df.shape[1]} columns")

if "Flood?" in df.columns:
    print("Flood column detected: ✅ YES")
else:
    print("Flood column detected: ❌ NO")

print("Dataset loaded: ✅ YES")

print("\nNext stage:")
print("➡ Data cleaning and preprocessing")

print("=" * 70)