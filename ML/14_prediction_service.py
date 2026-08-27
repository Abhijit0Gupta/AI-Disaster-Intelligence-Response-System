from pathlib import Path
import pandas as pd
import joblib

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# REAL-TIME PREDICTION SERVICE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "final_flood_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("REAL-TIME FLOOD PREDICTION SERVICE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD MODEL AND PREPROCESSOR
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("\n✅ Final model loaded.")
print("✅ Preprocessor loaded.")

# ------------------------------------------------------------
# 2. USER INPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENTER WEATHER INFORMATION")
print("=" * 70)

station = input("\nStation Name: ")

year = int(input("Year: "))
month = int(input("Month (1-12): "))

max_temp = float(input("Maximum Temperature (°C): "))
min_temp = float(input("Minimum Temperature (°C): "))

rainfall = float(input("Rainfall (mm): "))

humidity = float(
    input("Relative Humidity (%): ")
)

wind_speed = float(
    input("Wind Speed: ")
)

cloud_coverage = float(
    input("Cloud Coverage: ")
)

bright_sunshine = float(
    input("Bright Sunshine: ")
)

latitude = float(
    input("Latitude: ")
)

longitude = float(
    input("Longitude: ")
)

altitude = float(
    input("Altitude: ")
)

# ------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------

temperature_range = max_temp - min_temp

humidity_rainfall_index = (
    humidity * rainfall
)

# Season
if month in [12, 1, 2]:
    season = "Winter"
elif month in [3, 4, 5]:
    season = "Pre-Monsoon"
elif month in [6, 7, 8, 9]:
    season = "Monsoon"
else:
    season = "Post-Monsoon"

# Rainfall category
if rainfall <= 50:
    rainfall_category = "Low"
elif rainfall <= 200:
    rainfall_category = "Moderate"
elif rainfall <= 500:
    rainfall_category = "High"
else:
    rainfall_category = "Extreme"

# ------------------------------------------------------------
# 4. CREATE INPUT DATAFRAME
# ------------------------------------------------------------

input_data = pd.DataFrame([{
    "Station_Names": station,
    "Year": year,
    "Month": month,
    "Max_Temp": max_temp,
    "Min_Temp": min_temp,
    "Rainfall": rainfall,
    "Relative_Humidity": humidity,
    "Wind_Speed": wind_speed,
    "Cloud_Coverage": cloud_coverage,
    "Bright_Sunshine": bright_sunshine,
    "LATITUDE": latitude,
    "LONGITUDE": longitude,
    "ALT": altitude,
    "Season": season,
    "Rainfall_Category": rainfall_category,
    "Temperature_Range": temperature_range,
    "Humidity_Rainfall_Index": humidity_rainfall_index
}])

# ------------------------------------------------------------
# 5. PREPROCESS INPUT
# ------------------------------------------------------------

X = preprocessor.transform(input_data)

# ------------------------------------------------------------
# 6. MAKE PREDICTION
# ------------------------------------------------------------

prediction = model.predict(X)[0]

probability = model.predict_proba(X)[0][1]

# ------------------------------------------------------------
# 7. DETERMINE RISK LEVEL
# ------------------------------------------------------------

if probability >= 0.80:
    risk_level = "HIGH"
elif probability >= 0.50:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

# ------------------------------------------------------------
# 8. DISPLAY RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FLOOD PREDICTION RESULT")
print("=" * 70)

print("\nStation:", station)

print(
    "Flood Prediction:",
    "FLOOD" if prediction == 1 else "NO FLOOD"
)

print(
    "Flood Probability:",
    f"{probability * 100:.2f}%"
)

print(
    "Risk Level:",
    risk_level
)

print("\n" + "=" * 70)

if risk_level == "HIGH":
    print("🚨 HIGH FLOOD RISK")
    print("Immediate preparedness and warning measures are recommended.")

elif risk_level == "MEDIUM":
    print("⚠️ MEDIUM FLOOD RISK")
    print("Continue monitoring weather conditions.")

else:
    print("✅ LOW FLOOD RISK")
    print("No immediate flood warning indicated.")

print("=" * 70)