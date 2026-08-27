from pathlib import Path
import pandas as pd

# ============================================================
# FLOOD PREDICTION - INDIA DATASET ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "India_Flood_Inventory_v3 (2).csv"
)

print("=" * 70)
print("INDIA FLOOD INVENTORY - DATASET ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{RAW_FILE}"
    )

df = pd.read_csv(RAW_FILE)

print("\nDataset loaded successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# 2. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

# ------------------------------------------------------------
# 3. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

if missing.empty:
    print("No missing values.")
else:
    print(missing.to_string())

# ------------------------------------------------------------
# 5. DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print(
    f"Duplicate rows: {df.duplicated().sum()}"
)

# ------------------------------------------------------------
# 6. UNIQUE STATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INDIAN STATES")
print("=" * 70)

if "State" in df.columns:

    print(
        df["State"]
        .value_counts(dropna=False)
        .to_string()
    )

# ------------------------------------------------------------
# 7. MAIN CAUSES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MAIN CAUSES")
print("=" * 70)

if "Main Cause" in df.columns:

    print(
        df["Main Cause"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

# ------------------------------------------------------------
# 8. SEVERITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEVERITY")
print("=" * 70)

if "Severity" in df.columns:

    print(
        df["Severity"]
        .value_counts(dropna=False)
        .to_string()
    )

# ------------------------------------------------------------
# 9. DATE INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

for column in ["Start Date", "End Date"]:

    if column in df.columns:

        dates = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        print(f"\n{column}")

        print(
            f"Valid dates : {dates.notna().sum()}"
        )

        print(
            f"Invalid dates : {dates.isna().sum()}"
        )

        if dates.notna().any():

            print(
                f"Minimum : {dates.min()}"
            )

            print(
                f"Maximum : {dates.max()}"
            )

# ------------------------------------------------------------
# 10. NUMERICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(
    df.describe(include="all").transpose()
    .to_string()
)

# ------------------------------------------------------------
# 11. FLOOD EVENT COUNTS BY STATE
# ------------------------------------------------------------

if "State" in df.columns:

    print("\n" + "=" * 70)
    print("TOP STATES BY FLOOD EVENTS")
    print("=" * 70)

    state_counts = (
        df["State"]
        .value_counts()
        .head(20)
    )

    print(state_counts.to_string())

# ------------------------------------------------------------
# 12. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This dataset contains recorded flood events."
)

print(
    "It does not contain a direct Flood? 0/1 target."
)

print(
    "Therefore, a flood-occurrence classifier cannot "
    "be honestly trained from this file alone."
)

print(
    "\nNext stage:"
)

print(
    "Clean the Indian flood inventory and prepare "
    "a valid prediction target."
)