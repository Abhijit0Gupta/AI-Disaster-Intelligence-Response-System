import streamlit as st
import pandas as pd
import requests
from datetime import datetime
API_URL = "http://127.0.0.1:8000/analyze"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Disaster Intelligence System",
    page_icon="🚨",
    layout="wide"
)

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .risk-critical {
        padding: 20px;
        border-radius: 12px;
        background-color: #4b252b;
        color: #ff6b6b;
        font-size: 22px;
        font-weight: 700;
        margin: 20px 0;
    }

    .risk-high {
        padding: 20px;
        border-radius: 12px;
        background-color: #493b20;
        color: #ffd166;
        font-size: 22px;
        font-weight: 700;
        margin: 20px 0;
    }

    .risk-moderate {
        padding: 20px;
        border-radius: 12px;
        background-color: #243c45;
        color: #66d9ef;
        font-size: 22px;
        font-weight: 700;
        margin: 20px 0;
    }

    .risk-low {
        padding: 20px;
        border-radius: 12px;
        background-color: #173b2a;
        color: #69db7c;
        font-size: 22px;
        font-weight: 700;
        margin: 20px 0;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .small-note {
        opacity: 0.7;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(value, maximum))


def calculate_risk(
    disaster_type,
    rainfall,
    water_level,
    affected_population,
    damage_percentage
):
    """
    Explainable disaster risk engine.

    The score combines:
    - rainfall severity
    - water level
    - affected population
    - infrastructure/property damage
    - disaster type
    """

    rainfall_score = clamp((rainfall / 200) * 25)

    water_score = clamp((water_level / 5) * 25)

    population_score = clamp((affected_population / 10000) * 20)

    damage_score = clamp((damage_percentage / 100) * 25)

    disaster_bonus = {
        "Flood": 5,
        "Cyclone": 5,
        "Earthquake": 7,
        "Landslide": 6
    }

    type_score = disaster_bonus.get(disaster_type, 0)

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

    return score, level


def calculate_damage(damage_percentage, affected_population):
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

    return infrastructure_damage, estimated_affected_assets, severity


def calculate_transport_risk(
    rainfall,
    water_level,
    damage_percentage
):
    score = (
        (rainfall / 200) * 40
        + (water_level / 5) * 35
        + (damage_percentage / 100) * 25
    )

    return round(clamp(score), 1)


def calculate_facility_risk(
    affected_population,
    damage_percentage,
    water_level
):
    score = (
        (affected_population / 10000) * 40
        + (damage_percentage / 100) * 35
        + (water_level / 5) * 25
    )

    return round(clamp(score), 1)


def generate_actions(disaster_type, risk_level):
    if risk_level == "CRITICAL":
        return [
            "🚨 Activate emergency response procedures immediately.",
            "👥 Prepare evacuation routes and emergency shelters.",
            "📡 Monitor disaster conditions continuously.",
            "🚑 Deploy emergency response and medical teams.",
            "🏠 Assess infrastructure and property damage.",
            "📦 Prioritize essential resources for high-impact areas."
        ]

    elif risk_level == "HIGH":
        return [
            "⚠️ Activate enhanced disaster monitoring.",
            "👥 Prepare evacuation and shelter facilities.",
            "📡 Monitor rainfall, water levels and field conditions.",
            "🚑 Keep emergency response teams on standby.",
            "📦 Pre-position essential relief resources."
        ]

    elif risk_level == "MODERATE":
        return [
            "📡 Continue regular disaster monitoring.",
            "👥 Prepare emergency resources.",
            "🏠 Inspect vulnerable infrastructure.",
            "📦 Maintain essential resource availability."
        ]

    else:
        return [
            "📡 Continue routine monitoring.",
            "🏠 Inspect vulnerable areas periodically.",
            "📦 Maintain basic emergency preparedness."
        ]


def calculate_resources(
    affected_population,
    risk_score
):
    """
    Priority-based resource allocation.
    """

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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🚨 AI Disaster Intelligence System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-powered disaster intelligence, damage assessment, '
    'priority-based resource allocation and emergency decision support.'
    '</p>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Controls")

    st.write("Enter disaster information and run the intelligence engine.")

    st.info(
        "This prototype uses an explainable decision engine. "
        "A trained ML model can be integrated later using historical disaster datasets."
    )

    if st.button("🔄 Reset Analysis"):
        st.session_state.clear()
        st.rerun()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🌍 Disaster Information</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    disaster_type = st.selectbox(
        "Disaster Type",
        [
            "Flood",
            "Earthquake",
            "Cyclone",
            "Landslide"
        ]
    )

    location = st.text_input(
        "Location",
        placeholder="Example: Bhubaneswar"
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=10.0
    )

with c2:

    water_level = st.number_input(
        "Water Level (m)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1
    )

    affected_population = st.number_input(
        "Affected Population",
        min_value=0,
        max_value=10000000,
        value=0,
        step=100
    )

    damage_percentage = st.number_input(
        "Damage Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0
    )


st.markdown("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Disaster",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not location.strip():

        st.error("Please enter a disaster location.")

        st.stop()

    if affected_population <= 0:

        st.error("Affected population must be greater than 0.")

        st.stop()
        payload = {
        "disaster_type": disaster_type,
        "location": location,
        "rainfall": rainfall,
        "water_level": water_level,
        "affected_population": affected_population,
        "damage_percentage": damage_percentage
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the backend. "
            "Make sure the FastAPI server is running on port 8000."
        )
        st.stop()

    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {e}")
        st.stop()    
    # --------------------------------------------------------
    # RISK ENGINE
    # --------------------------------------------------------

    risk_score = result["risk"]["score"]
    risk_level = result["risk"]["level"]
    priority = result["risk"]["priority"]

    # --------------------------------------------------------
    # DAMAGE ENGINE
    # --------------------------------------------------------

    infrastructure_damage, estimated_assets, damage_severity = (
        calculate_damage(
            damage_percentage,
            affected_population
        )
    )

    # --------------------------------------------------------
    # TRANSPORT RISK
    # --------------------------------------------------------

    transport_risk = calculate_transport_risk(
        rainfall,
        water_level,
        damage_percentage
    )

    # --------------------------------------------------------
    # FACILITY RISK
    # --------------------------------------------------------

    facility_risk = calculate_facility_risk(
        affected_population,
        damage_percentage,
        water_level
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
        affected_population,
        risk_score
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if risk_score >= 70:
        priority = "P1 - IMMEDIATE"
    elif risk_score >= 50:
        priority = "P2 - HIGH"
    elif risk_score >= 30:
        priority = "P3 - MODERATE"
    else:
        priority = "P4 - LOW"

    # --------------------------------------------------------
    # DATA SOURCES
    # --------------------------------------------------------

    data_sources = [
        "🌧️ Rainfall / Weather Data",
        "🌊 Water-Level / Sensor Data",
        "👥 Population Impact Data",
        "🏠 Damage Assessment Data",
        "🚑 Emergency Resource Data"
    ]

    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Disaster Intelligence Report</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Analysis Location:** {location}"
    )

    st.write(
        f"**Disaster Type:** {disaster_type}"
    )

    st.write(
        f"**Analysis Time:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

    # ========================================================
    # KEY METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    with m2:
        st.metric(
            "Priority",
            priority
        )

    with m3:
        st.metric(
            "Affected Population",
            f"{affected_population:,}"
        )

    with m4:
        st.metric(
            "Damage",
            f"{damage_percentage:.1f}%"
        )

    # ========================================================
    # RISK ALERT
    # ========================================================

    if risk_level == "CRITICAL":

        st.markdown(
            '<div class="risk-critical">'
            '🚨 CRITICAL RISK: Immediate emergency response is recommended.'
            '</div>',
            unsafe_allow_html=True
        )

    elif risk_level == "HIGH":

        st.markdown(
            '<div class="risk-high">'
            '⚠️ HIGH RISK: Emergency preparedness and rapid response are required.'
            '</div>',
            unsafe_allow_html=True
        )

    elif risk_level == "MODERATE":

        st.markdown(
            '<div class="risk-moderate">'
            '🟡 MODERATE RISK: Increased monitoring and preparedness recommended.'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '🟢 LOW RISK: Continue routine monitoring and preparedness.'
            '</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # AI DECISION EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 AI Decision Explanation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The intelligence engine calculated the priority using multiple "
        "disaster indicators rather than relying on a single parameter."
    )

    explanation_data = pd.DataFrame({
        "Risk Factor": [
            "Rainfall",
            "Water Level",
            "Affected Population",
            "Damage",
            "Disaster Type"
        ],
        "Observed Value": [
            f"{rainfall:.1f} mm",
            f"{water_level:.1f} m",
            f"{affected_population:,}",
            f"{damage_percentage:.1f}%",
            disaster_type
        ]
    })

    st.table(explanation_data)

    # ========================================================
    # DAMAGE ASSESSMENT
    # ========================================================

    st.markdown(
        '<div class="section-title">🏚️ Rapid Damage Assessment</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Damage Severity",
            damage_severity
        )

    with d2:
        st.metric(
            "Infrastructure Damage",
            f"{infrastructure_damage:.1f}%"
        )

    with d3:
        st.metric(
            "Estimated Impacted Assets/People",
            f"{estimated_assets:,}"
        )

    # ========================================================
    # RISK BREAKDOWN
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Risk Factor Analysis</div>',
        unsafe_allow_html=True
    )

    rainfall_component = clamp((rainfall / 200) * 25)
    water_component = clamp((water_level / 5) * 25)
    population_component = clamp((affected_population / 10000) * 20)
    damage_component = clamp((damage_percentage / 100) * 25)

    risk_breakdown = pd.DataFrame({
        "Factor": [
            "Rainfall",
            "Water Level",
            "Population Impact",
            "Damage"
        ],
        "Risk Contribution": [
            rainfall_component,
            water_component,
            population_component,
            damage_component
        ]
    })

    st.bar_chart(
        risk_breakdown.set_index("Factor")
    )

    # ========================================================
    # INFRASTRUCTURE / TRANSPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">🚧 Infrastructure & Transport Risk</div>',
        unsafe_allow_html=True
    )

    t1, t2 = st.columns(2)

    with t1:

        st.metric(
            "🚗 Road / Transport Risk",
            f"{transport_risk}/100"
        )

        if transport_risk >= 70:
            st.error(
                "High transport disruption risk. "
                "Emergency routes should be assessed immediately."
            )
        elif transport_risk >= 40:
            st.warning(
                "Moderate transport disruption risk."
            )
        else:
            st.success(
                "Transport disruption risk currently low."
            )

    with t2:

        st.metric(
            "🏥 Critical Facility Risk",
            f"{facility_risk}/100"
        )

        if facility_risk >= 70:
            st.error(
                "Critical facilities may require priority assessment."
            )
        elif facility_risk >= 40:
            st.warning(
                "Critical facility monitoring recommended."
            )
        else:
            st.success(
                "Critical facility risk currently low."
            )

    # ========================================================
    # RESOURCE ALLOCATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📦 Priority-Based Resource Allocation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Resources are prioritized according to calculated disaster "
        "risk and affected population."
    )

    resource_data = pd.DataFrame({
        "Resource": [
            "🚑 Medical Units",
            "🚒 Rescue Teams",
            "🍱 Food Kits",
            "💧 Water Units",
            "🚑 Ambulances"
        ],
        "Recommended Quantity": [
            medical_units,
            rescue_teams,
            food_kits,
            water_units,
            ambulances
        ],
        "Priority": [
            priority,
            priority,
            priority,
            priority,
            priority
        ]
    })

    st.dataframe(
        resource_data,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "✅ Resources prioritized according to disaster risk "
        "and affected population."
    )

    # ========================================================
    # EMERGENCY ACTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">🛟 Recommended Emergency Actions</div>',
        unsafe_allow_html=True
    )

    actions = generate_actions(
        disaster_type,
        risk_level
    )

    for action in actions:
        st.write(action)

    # ========================================================
    # DATA INTEGRATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🔗 Integrated Disaster Data</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The system combines multiple disaster indicators into a "
        "single decision-support view."
    )

    source_cols = st.columns(5)

    for i, source in enumerate(data_sources):

        with source_cols[i]:

            st.info(source)

    # ========================================================
    # PRIORITY ZONE
    # ========================================================

    st.markdown(
        '<div class="section-title">📍 Response Priority</div>',
        unsafe_allow_html=True
    )

    if risk_score >= 70:

        st.error(
            f"🔴 {location} should receive FIRST-PRIORITY emergency resources."
        )

    elif risk_score >= 50:

        st.warning(
            f"🟠 {location} should receive HIGH-PRIORITY response resources."
        )

    elif risk_score >= 30:

        st.info(
            f"🟡 {location} should receive MODERATE-PRIORITY monitoring."
        )

    else:

        st.success(
            f"🟢 {location} currently falls under LOW response priority."
        )

    # ========================================================
    # RESOURCE CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Resource Distribution</div>',
        unsafe_allow_html=True
    )

    resource_chart = pd.DataFrame({
        "Resource": [
            "Medical",
            "Rescue",
            "Food",
            "Water",
            "Ambulance"
        ],
        "Quantity": [
            medical_units,
            rescue_teams,
            food_kits,
            water_units,
            ambulances
        ]
    })

    st.bar_chart(
        resource_chart.set_index("Resource")
    )

    # ========================================================
    # EXPORT REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📥 Export Decision Report</div>',
        unsafe_allow_html=True
    )

    report = f"""
AI DISASTER INTELLIGENCE SYSTEM
================================

Location: {location}
Disaster Type: {disaster_type}
Analysis Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

RISK ASSESSMENT
---------------
Risk Score: {risk_score}/100
Risk Level: {risk_level}
Priority: {priority}

DISASTER INDICATORS
-------------------
Rainfall: {rainfall:.1f} mm
Water Level: {water_level:.1f} m
Affected Population: {affected_population:,}
Damage: {damage_percentage:.1f}%

DAMAGE ASSESSMENT
-----------------
Damage Severity: {damage_severity}
Estimated Impact: {estimated_assets:,}

INFRASTRUCTURE
--------------
Transport Risk: {transport_risk}/100
Critical Facility Risk: {facility_risk}/100

RESOURCE ALLOCATION
-------------------
Medical Units: {medical_units}
Rescue Teams: {rescue_teams}
Food Kits: {food_kits}
Water Units: {water_units}
Ambulances: {ambulances}

EMERGENCY ACTIONS
-----------------
"""

    for action in actions:
        report += f"- {action}\n"

    report += """

DISCLAIMER
----------
This prototype is a decision-support system and should not replace
official emergency-management authorities or verified field data.
"""

    st.download_button(
        label="📄 Download Disaster Report",
        data=report,
        file_name="disaster_intelligence_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    st.divider()

    st.success(
        "✅ Disaster analysis completed successfully. "
        "Use the risk, damage, infrastructure and resource sections "
        "to support emergency decision-making."
    )

else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.info(
        "👆 Enter the disaster information above and click "
        "**Analyze Disaster** to generate the complete intelligence report."
    )

    st.markdown(
        '<div class="section-title">🚀 System Capabilities</div>',
        unsafe_allow_html=True
    )

    capabilities = pd.DataFrame({
        "Capability": [
            "Risk Assessment",
            "Rapid Damage Assessment",
            "Priority Classification",
            "Resource Allocation",
            "Transport Risk Analysis",
            "Critical Facility Risk",
            "Emergency Recommendations",
            "Decision Explanation",
            "Report Export"
        ],
        "Status": [
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅",
            "✅"
        ]
    })

    st.table(capabilities)