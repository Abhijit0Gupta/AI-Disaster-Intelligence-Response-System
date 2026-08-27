from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FloodPrediction.csv"

df = pd.read_csv(DATA_PATH)

# Keep only labeled records
labeled = df[df["Flood?"].notna()].copy()

print("=" * 70)
print("FLOOD LABEL INVESTIGATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. RAINFALL THRESHOLD ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RAINFALL ANALYSIS")
print("=" * 60)

for threshold in [50, 100, 150, 200, 250, 300, 400, 500, 600]:
    predicted = labeled["Rainfall"] >= threshold

    accuracy = (predicted == (labeled["Flood?"] == 1)).mean()

    print(
        f"Rainfall >= {threshold:3} mm -> "
        f"matches flood label: {accuracy * 100:.2f}%"
    )


# ------------------------------------------------------------
# 2. FLOOD RATE BY MONTH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FLOOD RATE BY MONTH")
print("=" * 60)

month_analysis = (
    labeled.groupby("Month")["Flood?"]
    .agg(["count", "sum", "mean"])
)

month_analysis["flood_percentage"] = month_analysis["mean"] * 100

print(month_analysis)


# ------------------------------------------------------------
# 3. FLOOD RATE BY STATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FLOOD RATE BY STATION")
print("=" * 60)

station_analysis = (
    labeled.groupby("Station_Names")["Flood?"]
    .agg(["count", "sum", "mean"])
)

station_analysis["flood_percentage"] = (
    station_analysis["mean"] * 100
)

print(
    station_analysis.sort_values(
        "flood_percentage",
        ascending=False
    )
)


# ------------------------------------------------------------
# 4. FLOOD RATE BY RAINFALL RANGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FLOOD RATE BY RAINFALL RANGE")
print("=" * 60)

labeled["rainfall_range"] = pd.cut(
    labeled["Rainfall"],
    bins=[-1, 50, 100, 200, 300, 500, 1000, float("inf")],
    labels=[
        "0-50",
        "51-100",
        "101-200",
        "201-300",
        "301-500",
        "501-1000",
        "1000+"
    ]
)

rainfall_analysis = (
    labeled.groupby("rainfall_range", observed=True)["Flood?"]
    .agg(["count", "sum", "mean"])
)

rainfall_analysis["flood_percentage"] = (
    rainfall_analysis["mean"] * 100
)

print(rainfall_analysis)


# ------------------------------------------------------------
# 5. FLOOD RATE BY YEAR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FLOOD RATE BY YEAR")
print("=" * 60)

year_analysis = (
    labeled.groupby("Year")["Flood?"]
    .agg(["count", "sum", "mean"])
)

year_analysis["flood_percentage"] = (
    year_analysis["mean"] * 100
)

print(year_analysis.to_string())


print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)