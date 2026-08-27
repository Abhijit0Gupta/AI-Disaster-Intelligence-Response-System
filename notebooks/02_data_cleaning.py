from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# INDIA FLOOD INVENTORY - DATA CLEANING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "India_Flood_Inventory_v3 (2).csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "India_Flood_Inventory_cleaned.csv"
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("INDIA FLOOD INVENTORY - DATA CLEANING")
print("=" * 70)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{RAW_FILE}"
    )

df = pd.read_csv(RAW_FILE)

print("\nDataset loaded successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("\nColumn names cleaned.")

# ============================================================
# 3. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("REMOVING COMPLETELY EMPTY COLUMNS")
print("=" * 70)

empty_columns = [
    column
    for column in df.columns
    if df[column].isna().all()
]

print("Completely empty columns:")

if empty_columns:
    for column in empty_columns:
        print(f"- {column}")

    df = df.drop(columns=empty_columns)

else:
    print("None")

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

if duplicates > 0:

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Removed {duplicates} duplicate rows."
    )

else:

    print("No duplicate rows found.")

# ============================================================
# 5. CLEAN DATE COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("DATE CLEANING")
print("=" * 70)

for column in ["Start Date", "End Date"]:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            dayfirst=True
        )

        print(
            f"{column}: "
            f"{df[column].notna().sum()} valid dates"
        )

# ============================================================
# 6. CREATE DATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATE FEATURES")
print("=" * 70)

if "Start Date" in df.columns:

    df["Start_Year"] = (
        df["Start Date"].dt.year
    )

    df["Start_Month"] = (
        df["Start Date"].dt.month
    )

    df["Start_Day"] = (
        df["Start Date"].dt.day
    )

    print("- Start_Year")
    print("- Start_Month")
    print("- Start_Day")

# ============================================================
# 7. CREATE VALID TARGET
# ============================================================

print("\n" + "=" * 70)
print("CREATING TARGET")
print("=" * 70)

TARGET_SOURCE = "Human fatality"

if TARGET_SOURCE not in df.columns:
    raise ValueError(
        f"Required column '{TARGET_SOURCE}' not found."
    )

# Convert fatalities to numeric
df[TARGET_SOURCE] = pd.to_numeric(
    df[TARGET_SOURCE],
    errors="coerce"
)

# Remove records where target cannot be determined
before_target = len(df)

df = df.dropna(
    subset=[TARGET_SOURCE]
).reset_index(drop=True)

removed_target = (
    before_target - len(df)
)

print(
    f"Records with missing fatality data removed: "
    f"{removed_target}"
)

# High-impact definition:
# 10 or more human fatalities
df["High_Impact"] = (
    df[TARGET_SOURCE] >= 10
).astype(int)

print("\nTarget definition:")
print(
    "High_Impact = 1 -> Human fatality >= 10"
)
print(
    "High_Impact = 0 -> Human fatality < 10"
)

print("\nTarget distribution:")

print(
    df["High_Impact"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")

print(
    (
        df["High_Impact"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

# ============================================================
# 7B. DEFINE TARGET / OUTCOME COLUMNS
# ============================================================

TARGET_COLUMN = "High_Impact"

OUTCOME_COLUMNS = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Extent of damage",
    "Area Affected"
]

print("\n" + "=" * 70)
print("OUTCOME / LEAKAGE COLUMNS")
print("=" * 70)

for column in OUTCOME_COLUMNS:

    if column in df.columns:
        print(f"EXCLUDE FROM MODEL: {column}")

# ============================================================
# 8. CLEAN CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL DATA CLEANING")
print("=" * 70)

categorical_columns = [
    "Main Cause",
    "Districts",
    "State"
]

for column in categorical_columns:

    if column not in df.columns:
        continue

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

    df[column] = (
        df[column]
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA
            }
        )
    )

    missing = df[column].isna().sum()

    print(
        f"{column}: {missing} missing values"
    )

# ============================================================
# 9. CLEAN MAIN CAUSE
# ============================================================

if "Main Cause" in df.columns:

    df["Main Cause"] = (
        df["Main Cause"]
        .str.lower()
        .str.strip()
    )

# ============================================================
# 10. NORMALIZE STATE NAMES
# ============================================================

if "State" in df.columns:

    df["State"] = (
        df["State"]
        .str.replace(
            "Jammu & Kashmir",
            "Jammu and Kashmir",
            regex=False
        )
        .str.strip()
    )

# ============================================================
# 11. CONVERT DURATION
# ============================================================

if "Duration(Days)" in df.columns:

    df["Duration(Days)"] = pd.to_numeric(
        df["Duration(Days)"],
        errors="coerce"
    )

    # Replace impossible/non-positive durations
    df.loc[
        df["Duration(Days)"] <= 0,
        "Duration(Days)"
    ] = np.nan

# ============================================================
# 12. HANDLE NUMERIC MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC MISSING VALUES")
print("=" * 70)

numeric_columns = [
    "Duration(Days)",
    "State_Codes",
    "District_LGD_Codes"
]

for column in numeric_columns:

    if column not in df.columns:
        continue

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    missing = df[column].isna().sum()

    print(
        f"{column}: {missing} missing"
    )

# ============================================================
# 13. REMOVE LEAKAGE-PRONE TARGET SOURCE
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CONTROL")
print("=" * 70)

print(
    "Human fatality will NOT be used as a model feature."
)

# Keep Human fatality in cleaned data for documentation,
# but model training will explicitly exclude it.

# ============================================================
# 14. SAVE CLEANED DATA
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETE")
print("=" * 70)

print(
    f"Rows remaining : {len(df)}"
)

print(
    f"Columns        : {len(df.columns)}"
)

print(
    f"Saved to       : {OUTPUT_FILE}"
)

print("\nFinal target distribution:")

print(
    df["High_Impact"]
    .value_counts()
    .sort_index()
)

print("\nNext stage:")
print(
    "Feature Engineering"
)