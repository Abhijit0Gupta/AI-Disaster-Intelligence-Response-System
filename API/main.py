from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

# ============================================================
# STATION METADATA
# ============================================================

STATIONS = {
    "Barisal": {"latitude": 22.70, "longitude": 90.3600, "alt": 4},
    "Bhola": {"latitude": 22.70, "longitude": 90.6600, "alt": 5},
    "Bogra": {"latitude": 24.88, "longitude": 89.3600, "alt": 20},
    "Chandpur": {"latitude": 23.26, "longitude": 90.6700, "alt": 7},
    "Chittagong (City-Ambagan)": {"latitude": 22.35, "longitude": 91.8166, "alt": 0},
    "Chittagong (IAP-Patenga)": {"latitude": 22.34, "longitude": 91.7900, "alt": 6},
    "Comilla": {"latitude": 23.48, "longitude": 91.1900, "alt": 10},
    "Cox's Bazar": {"latitude": 21.46, "longitude": 91.9800, "alt": 4},
    "Dhaka": {"latitude": 23.78, "longitude": 90.3900, "alt": 9},
    "Dinajpur": {"latitude": 25.63, "longitude": 88.6600, "alt": 37},
    "Faridpur": {"latitude": 23.61, "longitude": 89.8400, "alt": 9},
    "Feni": {"latitude": 23.01, "longitude": 91.3700, "alt": 8},
    "Hatiya": {"latitude": 22.29, "longitude": 91.1300, "alt": 4},
    "Ishurdi": {"latitude": 24.12, "longitude": 89.0400, "alt": 14},
    "Jessore": {"latitude": 23.17, "longitude": 89.2200, "alt": 7},
    "Khepupara": {"latitude": 21.98, "longitude": 90.2200, "alt": 3},
    "Khulna": {"latitude": 22.80, "longitude": 89.5800, "alt": 4},
    "Kutubdia": {"latitude": 21.83, "longitude": 91.8400, "alt": 6},
    "Madaripur": {"latitude": 23.17, "longitude": 90.1800, "alt": 5},
    "Maijdee Court": {"latitude": 22.83, "longitude": 91.0800, "alt": 6},
    "Mongla": {"latitude": 22.43, "longitude": 89.6600, "alt": 4},
    "Mymensingh": {"latitude": 24.75, "longitude": 90.4100, "alt": 19},
    "Patuakhali": {"latitude": 22.36, "longitude": 90.3400, "alt": 3},
    "Rajshahi": {"latitude": 24.35, "longitude": 88.5600, "alt": 20},
    "Rangamati": {"latitude": 22.67, "longitude": 92.2000, "alt": 63},
    "Rangpur": {"latitude": 25.72, "longitude": 89.2600, "alt": 34},
    "Sandwip": {"latitude": 22.50, "longitude": 91.4600, "alt": 6},
    "Satkhira": {"latitude": 22.68, "longitude": 89.0700, "alt": 6},
    "Sitakunda": {"latitude": 22.64, "longitude": 91.6400, "alt": 4},
    "Srimangal": {"latitude": 24.29, "longitude": 91.7300, "alt": 23},
    "Sylhet": {"latitude": 24.88, "longitude": 91.9300, "alt": 35},
    "Tangail": {"latitude": 24.15, "longitude": 89.5500, "alt": 10},
    "Teknaf": {"latitude": 20.87, "longitude": 92.2600, "alt": 4}
}

# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

MODEL_PATH = MODEL_DIR / "final_flood_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Disaster Intelligence & Response System",
    description="Flood prediction API using Machine Learning",
    version="1.0.0"
)

# ============================================================
# INPUT DATA MODEL
# ============================================================

class WeatherInput(BaseModel):
    station_name: str
    year: int
    month: int
    max_temp: float
    min_temp: float
    rainfall: float
    relative_humidity: float
    wind_speed: float
    cloud_coverage: float
    bright_sunshine: float

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Disaster Intelligence & Response System",
        "model": "Optimized Random Forest"
    }

# ============================================================
# STATION LIST
# ============================================================

@app.get("/stations")
def get_stations():
    return {
        "stations": list(STATIONS.keys())
    }

# ============================================================
# FLOOD PREDICTION
# ============================================================

@app.post("/predict")
def predict_flood(data: WeatherInput):

    # --------------------------------------------------------
    # Validate station
    # --------------------------------------------------------

    if data.station_name not in STATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown station: {data.station_name}"
        )

    # --------------------------------------------------------
    # Validate month
    # --------------------------------------------------------

    if data.month < 1 or data.month > 12:
        raise HTTPException(
            status_code=400,
            detail="Month must be between 1 and 12."
        )

    # --------------------------------------------------------
    # Get station metadata
    # --------------------------------------------------------

    station = STATIONS[data.station_name]

    latitude = station["latitude"]
    longitude = station["longitude"]
    altitude = station["alt"]

    # --------------------------------------------------------
    # Determine season
    # --------------------------------------------------------

    if data.month in [12, 1, 2]:
        season = "Winter"
    elif data.month in [3, 4]:
        season = "Pre-Monsoon"
    elif data.month in [5, 6, 7, 8, 9, 10]:
        season = "Monsoon"
    else:
        season = "Post-Monsoon"

    # --------------------------------------------------------
    # Determine rainfall category
    # --------------------------------------------------------

    if data.rainfall <= 50:
        rainfall_category = "Low"
    elif data.rainfall <= 200:
        rainfall_category = "Moderate"
    elif data.rainfall <= 500:
        rainfall_category = "High"
    else:
        rainfall_category = "Extreme"

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    temperature_range = data.max_temp - data.min_temp

    humidity_rainfall_index = (
        data.relative_humidity * data.rainfall
    )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "Station_Names": data.station_name,
        "Year": data.year,
        "Month": data.month,
        "Max_Temp": data.max_temp,
        "Min_Temp": data.min_temp,
        "Rainfall": data.rainfall,
        "Relative_Humidity": data.relative_humidity,
        "Wind_Speed": data.wind_speed,
        "Cloud_Coverage": data.cloud_coverage,
        "Bright_Sunshine": data.bright_sunshine,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "ALT": altitude,
        "Season": season,
        "Rainfall_Category": rainfall_category,
        "Temperature_Range": temperature_range,
        "Humidity_Rainfall_Index": humidity_rainfall_index
    }])

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_data = preprocessor.transform(input_data)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # --------------------------------------------------------
    # Recommended response
    # --------------------------------------------------------

    if risk_level == "HIGH":
        response = (
            "Immediate preparedness and flood-warning "
            "measures are recommended."
        )
    elif risk_level == "MEDIUM":
        response = (
            "Monitor weather conditions and prepare "
            "for possible flooding."
        )
    else:
        response = (
            "No immediate flood response required. "
            "Continue monitoring weather conditions."
        )

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        "station": data.station_name,
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude
        },
        "weather": {
            "rainfall_mm": data.rainfall,
            "humidity_percent": data.relative_humidity,
            "season": season
        },
        "prediction": {
            "flood": bool(prediction),
            "flood_probability": round(float(probability), 4),
            "risk_level": risk_level
        },
        "recommended_response": response
    }