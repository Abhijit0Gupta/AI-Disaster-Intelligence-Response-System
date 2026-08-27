from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "India_Flood_Inventory_features.csv"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "models"
    / "india_feature_importance.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = (
    RESULTS_DIR
    / "india_feature_validity_summary.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "High_Impact"


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - FEATURE VALIDITY ANALYSIS")
print("=" * 70)


# ============================================================
# 4. CHECK FILES
# ============================================================

print("\nChecking required files...")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )

if not IMPORTANCE_FILE.exists():

    raise FileNotFoundError(
        f"Feature importance file not found:\n{IMPORTANCE_FILE}"
    )

print("Feature dataset found.")
print("Feature importance file found.")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    FEATURE_FILE
)

importance_df = pd.read_csv(
    IMPORTANCE_FILE
)

print("\nDataset loaded successfully!")

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)


# ============================================================
# 6. TARGET CHECK
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


print(
    "\nTarget column:",
    TARGET
)

print(
    "\nTarget definition:"
)

print(
    "High_Impact = 1 -> Human fatality >= 10"
)

print(
    "High_Impact = 0 -> Human fatality < 10"
)


# ============================================================
# 7. FEATURE CATEGORIES
# ============================================================

VALID_FEATURES = {

    "Duration(Days)",
    "Main Cause",
    "State",
    "State_Codes",
    "District_LGD_Codes",
    "Start_Year",
    "Start_Month",
    "Start_Day",
    "Year",
    "Month",
    "Day",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Long_Event",
    "Cause_Group",
    "State_Count",
    "District_Count"

}


CAUTION_FEATURES = {

    "Event_Source_Available"

}


INVALID_FEATURES = {

    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Extent of damage",
    "Total_Human_Impact",
    "Reported_Human_Impact"

}


REVIEW_FEATURES = {

    "Districts",
    "Event Source"

}


# ============================================================
# 8. CREATE FEATURE VALIDITY TABLE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDITY RULES")
print("=" * 70)


records = []


for feature in df.columns:

    if feature == TARGET:

        continue


    if feature in INVALID_FEATURES:

        validity = "INVALID"

        reason = (
            "Directly describes disaster casualties or outcomes "
            "and may cause target leakage."
        )


    elif feature in CAUTION_FEATURES:

        validity = "CAUTION"

        reason = (
            "May depend on information availability after "
            "event reporting. Verify operational availability."
        )


    elif feature in REVIEW_FEATURES:

        validity = "REVIEW"

        reason = (
            "High-cardinality or reporting-related field. "
            "Requires manual review before operational use."
        )


    elif feature in VALID_FEATURES:

        validity = "VALID"

        reason = (
            "Can reasonably be used as a temporal, geographical, "
            "event, or cause-related predictor."
        )


    else:

        validity = "REVIEW"

        reason = (
            "Feature is not covered by the predefined "
            "validity rules and requires manual review."
        )


    records.append({

        "Original_Feature": feature,

        "Validity": validity,

        "Reason": reason

    })


validity_df = pd.DataFrame(
    records
)


# ============================================================
# 9. PROCESS FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING FEATURE IMPORTANCE")
print("=" * 70)


print(
    "\nImportance entries found:",
    len(importance_df)
)


# The model uses OneHotEncoder.
# Therefore feature names look like:
#
# numeric__District_Count
# numeric__Year
# categorical__State_Uttar Pradesh
#
# We extract the original feature name before
# matching it to the validity rules.


importance_records = []


for _, row in importance_df.iterrows():

    encoded_name = str(
        row["Feature"]
    )

    importance = float(
        row["Importance"]
    )


    if encoded_name.startswith(
        "numeric__"
    ):

        original_feature = (
            encoded_name
            .replace(
                "numeric__",
                "",
                1
            )
        )

        importance_records.append({

            "Encoded_Feature": encoded_name,

            "Original_Feature":
                original_feature,

            "Importance":
                importance

        })


    elif encoded_name.startswith(
        "categorical__"
    ):

        categorical_name = (
            encoded_name
            .replace(
                "categorical__",
                "",
                1
            )
        )


        # Match categorical feature names.
        matched_feature = None


        categorical_features = [

            "Main Cause",
            "State",
            "Cause_Group"

        ]


        for candidate in categorical_features:

            prefix = candidate + "_"

            if categorical_name.startswith(
                prefix
            ):

                matched_feature = candidate

                break


        if matched_feature is None:

            matched_feature = categorical_name


        importance_records.append({

            "Encoded_Feature": encoded_name,

            "Original_Feature":
                matched_feature,

            "Importance":
                importance

        })


    else:

        importance_records.append({

            "Encoded_Feature": encoded_name,

            "Original_Feature":
                encoded_name,

            "Importance":
                importance

        })


processed_importance = pd.DataFrame(
    importance_records
)


# ============================================================
# 10. AGGREGATE ONE-HOT IMPORTANCE
# ============================================================

print(
    "\nAggregating one-hot encoded features..."
)


aggregated_importance = (
    processed_importance
    .groupby(
        "Original_Feature",
        as_index=False
    )["Importance"]
    .sum()
)


aggregated_importance = (
    aggregated_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nAggregated feature importance:"
)

print(
    aggregated_importance
    .to_string(
        index=False
    )
)


# ============================================================
# 11. MERGE VALIDITY + IMPORTANCE
# ============================================================

validity_df = validity_df.merge(

    aggregated_importance,

    left_on="Original_Feature",

    right_on="Original_Feature",

    how="left"

)


validity_df["Importance"] = (
    validity_df["Importance"]
    .fillna(0)
)


validity_df = (
    validity_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDITY RESULTS")
print("=" * 70)


print(
    validity_df[
        [
            "Original_Feature",
            "Importance",
            "Validity",
            "Reason"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. VALIDITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDITY SUMMARY")
print("=" * 70)


summary_counts = (
    validity_df["Validity"]
    .value_counts()
)


for category in [

    "VALID",
    "CAUTION",
    "INVALID",
    "REVIEW"

]:

    count = summary_counts.get(
        category,
        0
    )

    print(
        f"{category:<10}: {count}"
    )


# ============================================================
# 14. TOP IMPORTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 70)


top_features = (
    validity_df
    .head(20)
)


print(
    top_features[
        [
            "Original_Feature",
            "Importance",
            "Validity"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 15. HIGH-IMPORTANCE FEATURE REVIEW
# ============================================================

print("\n" + "=" * 70)
print("HIGH-IMPORTANCE FEATURE REVIEW")
print("=" * 70)


high_importance = validity_df[
    validity_df["Importance"] >= 0.03
]


print(
    "\nFeatures with importance >= 0.03:",
    len(high_importance)
)


if high_importance.empty:

    print(
        "No features crossed the 0.03 threshold."
    )

else:

    print(
        high_importance[
            [
                "Original_Feature",
                "Importance",
                "Validity"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 16. PROXY LEAKAGE REVIEW
# ============================================================

print("\n" + "=" * 70)
print("PROXY LEAKAGE REVIEW")
print("=" * 70)


proxy_features = {

    "Duration(Days)": (
        "Longer events may be associated with greater "
        "impact, although duration is not directly derived "
        "from fatalities."
    ),

    "District_Count": (
        "Number of affected districts may represent "
        "event scale and therefore requires interpretation."
    ),

    "State_Count": (
        "Number of affected states may represent "
        "event scale and requires interpretation."
    ),

    "District_LGD_Codes": (
        "Administrative codes identify geography but may "
        "allow the model to learn location-specific patterns."
    ),

    "State_Codes": (
        "Administrative codes identify geography but may "
        "allow the model to learn location-specific patterns."
    ),

    "Year": (
        "Year may capture historical reporting, climate, "
        "infrastructure, or other temporal effects."
    )

}


for feature, reason in proxy_features.items():

    matching = validity_df[
        validity_df["Original_Feature"]
        == feature
    ]


    if not matching.empty:

        row = matching.iloc[0]


        print(
            f"\n{feature}"
        )

        print(
            f"Importance : {row['Importance']:.6f}"
        )

        print(
            "Validity   :",
            row["Validity"]
        )

        print(
            "Reason     :",
            reason
        )


# ============================================================
# 17. INVALID FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DIRECT LEAKAGE CHECK")
print("=" * 70)


invalid_features = validity_df[
    validity_df["Validity"]
    == "INVALID"
]


if invalid_features.empty:

    print(
        "\nPASS: No direct outcome leakage "
        "features are present."
    )

else:

    print(
        "\nWARNING: Invalid features detected!"
    )

    print(
        invalid_features[
            [
                "Original_Feature",
                "Importance"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 18. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE RECOMMENDATION")
print("=" * 70)


valid_count = (
    validity_df["Validity"]
    == "VALID"
).sum()


caution_count = (
    validity_df["Validity"]
    == "CAUTION"
).sum()


invalid_count = (
    validity_df["Validity"]
    == "INVALID"
).sum()


review_count = (
    validity_df["Validity"]
    == "REVIEW"
).sum()


print(
    f"\nValid features   : {valid_count}"
)

print(
    f"Caution features: {caution_count}"
)

print(
    f"Invalid features: {invalid_count}"
)

print(
    f"Review features  : {review_count}"
)


print(
    "\nInterpretation:"
)

print(
    "The model contains no direct casualty variables "
    "according to the current leakage rules."
)

print(
    "However, highly important temporal or geographical "
    "features should be reviewed for proxy effects."
)

print(
    "Feature importance alone should NOT be used "
    "to decide whether a feature is valid."
)


# ============================================================
# 19. SAVE REPORT
# ============================================================

validity_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nFeature validity report saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 20. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDITY ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nThe analysis checked:"
)

print(
    "1. Direct outcome leakage"
)

print(
    "2. Potential proxy leakage"
)

print(
    "3. Feature importance"
)

print(
    "4. Real-world feature validity"
)

print(
    "\nNext stage:"
)

print(
    "Review the validated feature set before "
    "final model selection."
)

print("=" * 70)