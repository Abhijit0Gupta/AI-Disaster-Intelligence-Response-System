from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

app = FastAPI(
    title="AI Disaster Intelligence Backend",
    description="Backend API for the AI Integrated Disaster Response and Management System.",
    version="1.0.0"
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class DisasterInput(BaseModel):
    disaster_type: str
    location: str
    rainfall: float = Field(ge=0, le=1000)
    water_level: float = Field(ge=0, le=20)
    affected_population: int = Field(gt=0, le=10000000)
    damage_percentage: float = Field(ge=0, le=100)


class RiskResult(BaseModel):
    score: float
    level: str
    priority: str


class DamageResult(BaseModel):
    infrastructure_damage: float
    estimated_affected_assets: int
    severity: str


class ResourceResult(BaseModel):
    medical_units: int
    rescue_teams: int
    food_kits: int
    water_units: int
    ambulances: int


class DisasterAnalysisResponse(BaseModel):
    location: str
    disaster_type: str
    analysis_time: str

    risk: RiskResult
    damage: DamageResult

    transport_risk: float
    facility_risk: float

    risk_breakdown: Dict[str, float]

    resources: ResourceResult
    actions: List[str]

    data_sources: List[str]


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_DISASTERS = {
    "Flood": 5,
    "Cyclone": 5,
    "Earthquake": 7,
    "Landslide": 6
}

DATA_SOURCES = [
    "Rainfall / Weather Data",
    "Water-Level / Sensor Data",
    "Population Impact Data",
    "Damage Assessment Data",
    "Emergency Resource Data"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(value, maximum))


def calculate_risk(
    disaster_type: str,
    rainfall: float,
    water_level: float,
    affected_population: int,
    damage_percentage: float
):
    rainfall_score = clamp((rainfall / 200) * 25, 0, 25)
    water_score = clamp((water_level / 5) * 25, 0, 25)
    population_score = clamp((affected_population / 10000) * 20, 0, 20)
    damage_score = clamp((damage_percentage / 100) * 25, 0, 25)

    type_score = SUPPORTED_DISASTERS.get(disaster_type, 0)

    total = (
        rainfall_score
        + water_score
        + population_score
        + damage_score
        + type_score
    )

    score = round(clamp(total), 1)

    if score >= 70:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 30:
        level = "MODERATE"
    else:
        level = "LOW"

    if score >= 70:
        priority = "P1 - IMMEDIATE"
    elif score >= 50:
        priority = "P2 - HIGH"
    elif score >= 30:
        priority = "P3 - MODERATE"
    else:
        priority = "P4 - LOW"

    breakdown = {
        "rainfall": round(rainfall_score, 1),
        "water_level": round(water_score, 1),
        "population": round(population_score, 1),
        "damage": round(damage_score, 1),
        "disaster_type": float(type_score)
    }

    return score, level, priority, breakdown


def calculate_damage(damage_percentage: float, affected_population: int):
    infrastructure_damage = damage_percentage

    estimated_affected_assets = round(
        affected_population * (damage_percentage / 100)
    )

    if damage_percentage >= 60:
        severity = "Severe"
    elif damage_percentage >= 35:
        severity = "High"
    elif damage_percentage >= 15:
        severity = "Moderate"
    else:
        severity = "Low"

    return (
        round(infrastructure_damage, 1),
        estimated_affected_assets,
        severity
    )


def calculate_transport_risk(
    rainfall: float,
    water_level: float,
    damage_percentage: float
):
    score = (
        (rainfall / 200) * 40
        + (water_level / 5) * 35
        + (damage_percentage / 100) * 25
    )

    return round(clamp(score), 1)


def calculate_facility_risk(
    affected_population: int,
    damage_percentage: float,
    water_level: float
):
    score = (
        (affected_population / 10000) * 40
        + (damage_percentage / 100) * 35
        + (water_level / 5) * 25
    )

    return round(clamp(score), 1)


def calculate_resources(
    affected_population: int,
    risk_score: float
):
    multiplier = 1 + (risk_score / 100)

    medical_units = max(
        10,
        round((affected_population / 100) * multiplier)
    )

    rescue_teams = max(
        5,
        round((affected_population / 500) * multiplier)
    )

    food_kits = max(
        50,
        round(affected_population * 2 * multiplier)
    )

    water_units = max(
        50,
        round(affected_population * 3 * multiplier)
    )

    ambulances = max(
        2,
        round((affected_population / 1500) * multiplier)
    )

    return (
        medical_units,
        rescue_teams,
        food_kits,
        water_units,
        ambulances
    )


def generate_actions(disaster_type: str, risk_level: str):
    actions = {
        "CRITICAL": [
            "Activate emergency response procedures immediately.",
            "Prepare evacuation routes and emergency shelters.",
            "Monitor disaster conditions continuously.",
            "Deploy emergency response and medical teams.",
            "Assess infrastructure and property damage.",
            "Prioritize essential resources for high-impact areas."
        ],
        "HIGH": [
            "Activate enhanced disaster monitoring.",
            "Prepare evacuation and shelter facilities.",
            "Monitor rainfall, water levels and field conditions.",
            "Keep emergency response teams on standby.",
            "Pre-position essential relief resources."
        ],
        "MODERATE": [
            "Continue regular disaster monitoring.",
            "Prepare emergency resources.",
            "Inspect vulnerable infrastructure.",
            "Maintain essential resource availability."
        ],
        "LOW": [
            "Continue routine monitoring.",
            "Inspect vulnerable areas periodically.",
            "Maintain basic emergency preparedness."
        ]
    }

    return actions.get(risk_level, actions["LOW"])


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Disaster Intelligence Backend",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/analyze", response_model=DisasterAnalysisResponse)
def analyze_disaster(data: DisasterInput):

    if data.disaster_type not in SUPPORTED_DISASTERS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported disaster type. "
                "Supported types: Flood, Earthquake, Cyclone, Landslide."
            )
        )

    if not data.location.strip():
        raise HTTPException(
            status_code=400,
            detail="Disaster location cannot be empty."
        )

    # --------------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------------

    (
        risk_score,
        risk_level,
        priority,
        risk_breakdown
    ) = calculate_risk(
        data.disaster_type,
        data.rainfall,
        data.water_level,
        data.affected_population,
        data.damage_percentage
    )

    # --------------------------------------------------------
    # DAMAGE ASSESSMENT
    # --------------------------------------------------------

    (
        infrastructure_damage,
        estimated_affected_assets,
        damage_severity
    ) = calculate_damage(
        data.damage_percentage,
        data.affected_population
    )

    # --------------------------------------------------------
    # TRANSPORT RISK
    # --------------------------------------------------------

    transport_risk = calculate_transport_risk(
        data.rainfall,
        data.water_level,
        data.damage_percentage
    )

    # --------------------------------------------------------
    # CRITICAL FACILITY RISK
    # --------------------------------------------------------

    facility_risk = calculate_facility_risk(
        data.affected_population,
        data.damage_percentage,
        data.water_level
    )

    # --------------------------------------------------------
    # RESOURCE ALLOCATION
    # --------------------------------------------------------

    (
        medical_units,
        rescue_teams,
        food_kits,
        water_units,
        ambulances
    ) = calculate_resources(
        data.affected_population,
        risk_score
    )

    # --------------------------------------------------------
    # EMERGENCY ACTIONS
    # --------------------------------------------------------

    actions = generate_actions(
        data.disaster_type,
        risk_level
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return DisasterAnalysisResponse(
        location=data.location.strip(),
        disaster_type=data.disaster_type,
        analysis_time=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

        risk=RiskResult(
            score=risk_score,
            level=risk_level,
            priority=priority
        ),

        damage=DamageResult(
            infrastructure_damage=infrastructure_damage,
            estimated_affected_assets=estimated_affected_assets,
            severity=damage_severity
        ),

        transport_risk=transport_risk,
        facility_risk=facility_risk,

        risk_breakdown=risk_breakdown,

        resources=ResourceResult(
            medical_units=medical_units,
            rescue_teams=rescue_teams,
            food_kits=food_kits,
            water_units=water_units,
            ambulances=ambulances
        ),

        actions=actions,
        data_sources=DATA_SOURCES
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
