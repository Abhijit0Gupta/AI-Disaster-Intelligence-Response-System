from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# INDIA FLOOD INVENTORY - FEATURE ENGINEERING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "India_Flood_Inventory_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "India_Flood_Inventory_features.csv"
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("INDIA FLOOD INVENTORY - FEATURE ENGINEERING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Rows    :", len(df))
print("Columns :", len(df.columns))

# ============================================================
# 2. CHECK TARGET
# ============================================================

TARGET = "High_Impact"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print("\nTarget column:", TARGET)

print("\nTarget distribution:")
print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

# ============================================================
# 3. DATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("DATE FEATURES")
print("=" * 70)

if "Start Date" in df.columns:

    df["Start Date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce"
    )

    df["Year"] = (
        df["Start Date"].dt.year
    )

    df["Month"] = (
        df["Start Date"].dt.month
    )

    df["Day"] = (
        df["Start Date"].dt.day
    )

    df["Quarter"] = (
        df["Start Date"].dt.quarter
    )

    print("- Year")
    print("- Month")
    print("- Day")
    print("- Quarter")

# ============================================================
# 4. CYCLICAL MONTH FEATURES
# ============================================================

print("\nCreating cyclical month features...")

if "Month" in df.columns:

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    print("- Month_Sin")
    print("- Month_Cos")

# ============================================================
# 5. DURATION FEATURES
# ============================================================

print("\nCreating duration features...")

if "Duration(Days)" in df.columns:

    df["Duration(Days)"] = pd.to_numeric(
        df["Duration(Days)"],
        errors="coerce"
    )

    df["Long_Event"] = (
        df["Duration(Days)"] >= 7
    ).astype(int)

    print("- Long_Event")

# ============================================================
# 6. CAUSE GROUPING
# ============================================================

print("\n" + "=" * 70)
print("CAUSE GROUPING")
print("=" * 70)

if "Main Cause" in df.columns:

    cause = (
        df["Main Cause"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["Cause_Group"] = "Other"

    df.loc[
        cause.str.contains("rain", na=False),
        "Cause_Group"
    ] = "Heavy Rain"

    df.loc[
        cause.str.contains("flash", na=False),
        "Cause_Group"
    ] = "Flash Flood"

    df.loc[
        cause.str.contains("landslide", na=False),
        "Cause_Group"
    ] = "Landslide"

    df.loc[
        cause.str.contains("dam", na=False),
        "Cause_Group"
    ] = "Dam/Reservoir"

    df.loc[
        cause.str.contains(
            "river|brahmaputra|tributary",
            na=False
        ),
        "Cause_Group"
    ] = "River/Water System"

    df.loc[
        cause.str.contains("cyclone", na=False),
        "Cause_Group"
    ] = "Cyclone"

    print("Cause groups created.")

    print("\nCause distribution:")
    print(
        df["Cause_Group"]
        .value_counts()
    )
# ============================================================
# 7. STATE / DISTRICT COUNTS
# ============================================================

print("\nCreating geographical count features...")

if "State" in df.columns:

    df["State_Count"] = (
        df["State"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(
            lambda values:
            sum(
                1
                for value in values
                if value.strip()
            )
        )
    )

    print("- State_Count")


if "Districts" in df.columns:

    df["District_Count"] = (
        df["Districts"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(
            lambda values:
            sum(
                1
                for value in values
                if value.strip()
            )
        )
    )

    print("- District_Count")

# ============================================================
# 8. EVENT SOURCE AVAILABILITY
# ============================================================

if "Event Source" in df.columns:

    df["Event_Source_Available"] = (
        df["Event Source"]
        .notna()
        .astype(int)
    )

    print("- Event_Source_Available")

# ============================================================
# 9. LEAKAGE CONTROL
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CONTROL")
print("=" * 70)

LEAKAGE_COLUMNS = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Extent of damage",
    "Area Affected",
    "Total_Human_Impact",
    "Reported_Human_Impact"
]

for column in LEAKAGE_COLUMNS:

    if column in df.columns:

        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            f"Removed leakage variable: {column}"
        )

print(
    "\nAll identified outcome/leakage "
    "variables excluded from model features."
)

# ============================================================
# 10. REMOVE IDENTIFIERS
# ============================================================

print("\n" + "=" * 70)
print("REMOVING IDENTIFIERS")
print("=" * 70)

identifier_columns = [
    "Unnamed: 0",
    "UEI",
    "Start Date",
    "End Date"
]

for column in identifier_columns:

    if column in df.columns:

        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            f"Removed: {column}"
        )

# ============================================================
# 11. CLEAN NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE CLEANING")
print("=" * 70)

numeric_columns = [
    "Duration(Days)",
    "State_Codes",
    "District_LGD_Codes",
    "Year",
    "Month",
    "Day",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Long_Event",
    "State_Count",
    "District_Count",
    "Event_Source_Available"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(
            f"{column}: numeric"
        )

# ============================================================
# 12. FINAL TARGET CHECK
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        "Target High_Impact was accidentally removed."
    )

# ============================================================
# 13. VERIFY NO LEAKAGE COLUMNS REMAIN
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEAKAGE VERIFICATION")
print("=" * 70)

remaining_leakage = [
    column
    for column in LEAKAGE_COLUMNS
    if column in df.columns
]

if remaining_leakage:

    print(
        "WARNING - leakage columns still present:"
    )

    for column in remaining_leakage:
        print("-", column)

else:

    print(
        "PASS: No identified leakage columns remain."
    )

# ============================================================
# 14. FINAL DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE DATA CHECK")
print("=" * 70)

print(
    "Rows    :", len(df)
)

print(
    "Columns :",
    len(df.columns)
)

print("\nFinal columns:")

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:>2}. {column}"
    )

print("\nMissing values:")

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:

    print("No missing values.")

else:

    print(missing)

# ============================================================
# 15. SAVE FEATURES
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)

print(
    "\nSaved to:"
)

print(OUTPUT_FILE)

print("\nNext stage:")
print("Model Training")