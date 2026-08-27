from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FloodPrediction.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("FLOOD LABEL ANALYSIS")
print("=" * 60)

# Only records having a flood label
labeled = df[df["Flood?"].notna()].copy()

print("\nLabeled records:", len(labeled))

print("\nFlood distribution:")
print(labeled["Flood?"].value_counts())

print("\nFlood distribution (%):")
print(labeled["Flood?"].value_counts(normalize=True) * 100)

print("\nLabeled records by station:")
print(
    labeled.groupby("Station_Names")["Flood?"]
    .agg(["count", "sum"])
)

print("\nLabeled records by year:")
print(
    labeled.groupby("Year")["Flood?"]
    .agg(["count", "sum"])
)

print("\nLabeled records by month:")
print(
    labeled.groupby("Month")["Flood?"]
    .agg(["count", "sum"])
)

print("\nRainfall statistics by flood label:")
print(
    labeled.groupby("Flood?")["Rainfall"]
    .agg(["count", "mean", "min", "max"])
)

print("\nSample labeled records:")
print(
    labeled[
        [
            "Station_Names",
            "Year",
            "Month",
            "Rainfall",
            "Relative_Humidity",
            "Flood?"
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)