from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
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
# 2. LOAD DATA
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
# 3. CHECK TARGET
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
# 4. DATE FEATURES
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
# 5. CYCLICAL MONTH FEATURES
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
# 6. DURATION FEATURES
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
# 7. CAUSE GROUPING
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
# 8. STATE COUNT
# ============================================================

print("\nCreating state count...")

if "State" in df.columns:

    df["State_Count"] = (
        df["State"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x:
            len([
                s for s in x.split(",")
                if s.strip()
            ])
        )
    )

    print("- State_Count")

# ============================================================
# 9. DISTRICT COUNT
# ============================================================

print("Creating district count...")

if "Districts" in df.columns:

    df["District_Count"] = (
        df["Districts"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x:
            len([
                d for d in x.split(",")
                if d.strip()
            ])
        )
    )

    print("- District_Count")

# ============================================================
# 10. EVENT SOURCE AVAILABILITY
# ============================================================

if "Event Source" in df.columns:

    df["Event_Source_Available"] = (
        df["Event Source"]
        .notna()
        .astype(int)
    )

    print("- Event_Source_Available")

# ============================================================
# 11. IMPORTANT LEAKAGE CONTROL
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CONTROL")
print("=" * 70)

leakage_columns = [
    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Total_Human_Impact",
    "Reported_Human_Impact"
]

for column in leakage_columns:

    if column in df.columns:

        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            f"Removed leakage variable: {column}"
        )

print(
    "\nHuman casualty information will NOT "
    "be used as model input."
)

# ============================================================
# 12. REMOVE IDENTIFIERS
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
            f"Removed identifier/date column: {column}"
        )

# ============================================================
# 13. CHECK TARGET AGAIN
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        "Target was accidentally removed."
    )

# ============================================================
# 14. CHECK DATA
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

print(
    "\nMissing values:"
)

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values.")

else:
    print(missing)

# ============================================================
# 15. SAVE
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