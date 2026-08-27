from pathlib import Path
import pandas as pd
import joblib

# ============================================================
# AI DISASTER INTELLIGENCE & RESPONSE SYSTEM
# ROBUST FLOOD PREDICTION SERVICE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "final_flood_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

print("=" * 70)
print("AI DISASTER INTELLIGENCE & RESPONSE SYSTEM")
print("ROBUST FLOOD PREDICTION SERVICE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD MODEL AND PREPROCESSOR
# ------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("\n✅ Final model loaded.")
    print("✅ Preprocessor loaded.")

except Exception as e:
    print("\n❌ Error loading model:", e)
    exit()

# ------------------------------------------------------------
# 2. STATION INFORMATION
# ------------------------------------------------------------

stations = {
    "Barisal": (22.7010, 90.3535),
    "Bhola": (22.6859, 90.6482),
    "Bogra": (24.8465, 89.3770),
    "Chandpur": (23.2330, 90.6712),
    "Chittagong (City-Ambagan)": (22.3569, 91.7832),
    "Chittagong (IAP-Patenga)": (22.2496, 91.8133),
    "Comilla": (23.4607, 91.1809),
    "Cox's Bazar": (21.4272, 92.0058),
    "Dhaka": (23.8103, 90.4125),
    "Dinajpur": (25.6279, 88.6332),
    "Faridpur": (23.6070, 89.8429),
    "Feni": (23.0159, 91.3976),
    "Hatiya": (22.2855, 91.1250),
    "Ishurdi": (24.1311, 89.0912),
    "Jessore": (23.1634, 89.2182),
    "Khepupara": (21.9970, 90.2220),
    "Khulna": (22.8456, 89.5403),
    "Kutubdia": (21.8167, 91.8583),
    "Madaripur": (23.1641, 90.1897),
    "Maijdee Court": (22.8696, 91.0994),
    "Mongla": (22.4833, 89.6000),
    "Mymensingh": (24.7471, 90.4203),
    "Patuakhali": (22.3596, 90.3299),
    "Rajshahi": (24.3745, 88.6042),
    "Rangamati": (22.6533, 92.1755),
    "Rangpur": (25.7439, 89.2752),
    "Sandwip": (22.4850, 91.4250),
    "Satkhira": (22.7185, 89.0705),
    "Sitakunda": (22.6150, 91.6850),
    "Srimangal": (24.3065, 91.7296),
    "Sylhet": (24.8949, 91.8687),
    "Tangail": (24.2513, 89.9167),
    "Teknaf": (20.8624, 92.3058)
}

# ------------------------------------------------------------
# 3. INPUT VALIDATION FUNCTIONS
# ------------------------------------------------------------

def get_number(prompt, minimum=None, maximum=None):

    while True:

        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"❌ Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"❌ Value must be at most {maximum}.")
                continue

            return value

        except ValueError:
            print("❌ Please enter a valid number.")


def get_integer(prompt, minimum=None, maximum=None):

    while True:

        try:
            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(f"❌ Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"❌ Value must be at most {maximum}.")
                continue

            return value

        except ValueError:
            print("❌ Please enter a valid integer.")


# ------------------------------------------------------------
# 4. STATION INPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SELECT WEATHER STATION")
print("=" * 70)

station_names = list(stations.keys())

for i, name in enumerate(station_names, 1):
    print(f"{i}. {name}")

while True:

    station_choice = get_integer(
        "\nEnter station number: ",
        1,
        len(station_names)
    )

    station = station_names[station_choice - 1]

    break

latitude, longitude = stations[station]

print("\nSelected station:", station)
print("Latitude:", latitude)
print("Longitude:", longitude)

# ------------------------------------------------------------
# 5. WEATHER INPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENTER WEATHER INFORMATION")
print("=" * 70)

year = get_integer(
    "Year: ",
    1948,
    2100
)

month = get_integer(
    "Month (1-12): ",
    1,
    12
)

max_temp = get_number(
    "Maximum Temperature (°C): ",
    -10,
    60
)

min_temp = get_number(
    "Minimum Temperature (°C): ",
    -20,
    50
)

while min_temp > max_temp:

    print("❌ Minimum temperature cannot exceed maximum temperature.")

    min_temp = get_number(
        "Minimum Temperature (°C): ",
        -20,
        50
    )

rainfall = get_number(
    "Rainfall (mm): ",
    0,
    5000
)

humidity = get_number(
    "Relative Humidity (%): ",
    0,
    100
)

wind_speed = get_number(
    "Wind Speed: ",
    0,
    200
)

cloud_coverage = get_number(
    "Cloud Coverage: ",
    0,
    100
)

bright_sunshine = get_number(
    "Bright Sunshine: ",
    0,
    24
)

altitude = get_number(
    "Altitude: ",
    -100,
    10000
)

# ------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------

temperature_range = max_temp - min_temp

humidity_rainfall_index = humidity * rainfall

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
# 7. CREATE INPUT DATAFRAME
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
# 8. PREPROCESS
# ------------------------------------------------------------

try:

    X = preprocessor.transform(input_data)

except Exception as e:

    print("\n❌ Preprocessing error:", e)
    exit()

# ------------------------------------------------------------
# 9. PREDICTION
# ------------------------------------------------------------

prediction = model.predict(X)[0]

probability = model.predict_proba(X)[0][1]

# ------------------------------------------------------------
# 10. RISK CLASSIFICATION
# ------------------------------------------------------------

if probability >= 0.80:

    risk_level = "HIGH"

    response = (
        "Immediate preparedness and flood-warning measures "
        "are recommended."
    )

elif probability >= 0.50:

    risk_level = "MEDIUM"

    response = (
        "Continue monitoring weather conditions and "
        "prepare for possible flooding."
    )

else:

    risk_level = "LOW"

    response = (
        "No immediate flood warning indicated. "
        "Continue routine monitoring."
    )

# ------------------------------------------------------------
# 11. DISPLAY RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FLOOD PREDICTION RESULT")
print("=" * 70)

print("\nStation:", station)

print("Location:")
print("Latitude :", latitude)
print("Longitude:", longitude)

print("\nWeather:")
print("Rainfall :", rainfall, "mm")
print("Humidity :", humidity, "%")
print("Season   :", season)

print("\nPrediction:")
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

print("\nRecommended Response:")
print(response)

print("\n" + "=" * 70)

if risk_level == "HIGH":

    print("🚨 HIGH FLOOD RISK")

elif risk_level == "MEDIUM":

    print("⚠️ MEDIUM FLOOD RISK")

else:

    print("✅ LOW FLOOD RISK")

print("=" * 70)

# ------------------------------------------------------------
# 12. SAVE PREDICTION
# ------------------------------------------------------------

prediction_output = MODEL_DIR / "latest_prediction.csv"

output = input_data.copy()

output["Flood_Prediction"] = prediction
output["Flood_Probability"] = probability
output["Risk_Level"] = risk_level
output["Recommended_Response"] = response

output.to_csv(
    prediction_output,
    index=False
)

print("\nPrediction saved to:")
print(prediction_output)

print("\n" + "=" * 70)
print("ROBUST PREDICTION SERVICE COMPLETE")
print("=" * 70)