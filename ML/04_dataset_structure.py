from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FloodPrediction.csv"

df = pd.read_csv(DATA_PATH)

labeled = df[df["Flood?"].notna()].copy()

print("=" * 70)
print("DATASET STRUCTURE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. STATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STATION ANALYSIS")
print("=" * 60)

print("Total stations:", df["Station_Names"].nunique())

print("\nRecords per station:")
print(df["Station_Names"].value_counts())


# ------------------------------------------------------------
# 2. YEAR RANGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TIME RANGE")
print("=" * 60)

print("Minimum year:", df["Year"].min())
print("Maximum year:", df["Year"].max())

print("Total years:", df["Year"].nunique())


# ------------------------------------------------------------
# 3. RECORDS PER YEAR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RECORDS PER YEAR")
print("=" * 60)

print(df["Year"].value_counts().sort_index().to_string())


# ------------------------------------------------------------
# 4. RECORDS PER STATION PER YEAR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STATION-YEAR COVERAGE")
print("=" * 60)

station_year = (
    df.groupby(["Station_Names", "Year"])
    .size()
)

print("Station-Year combinations:", len(station_year))

print("\nRecords per Station-Year:")
print(station_year.value_counts().sort_index())


# ------------------------------------------------------------
# 5. LABELED DATA TIME RANGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LABELED DATA TIME RANGE")
print("=" * 60)

print("Labeled minimum year:", labeled["Year"].min())
print("Labeled maximum year:", labeled["Year"].max())

print("\nLabeled records per year:")
print(
    labeled["Year"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 6. LABELED RECORDS BY STATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LABELED RECORDS BY STATION")
print("=" * 60)

print(
    labeled["Station_Names"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 7. LABELLED DATASET DATE RANGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LABELED DATA PERIOD")
print("=" * 60)

print(
    "Earliest:",
    labeled[["Year", "Month"]].sort_values(
        ["Year", "Month"]
    ).iloc[0].to_dict()
)

print(
    "Latest:",
    labeled[["Year", "Month"]].sort_values(
        ["Year", "Month"]
    ).iloc[-1].to_dict()
)


print("\n" + "=" * 70)
print("STRUCTURE ANALYSIS COMPLETE")
print("=" * 70)