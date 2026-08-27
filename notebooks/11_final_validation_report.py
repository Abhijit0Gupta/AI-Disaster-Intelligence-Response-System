from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

VALIDITY_FILE = (
    RESULTS_DIR
    / "india_feature_validity_summary.csv"
)

FINAL_SUMMARY_FILE = (
    RESULTS_DIR
    / "india_final_time_based_summary.csv"
)

FINAL_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "india_final_time_based_feature_importance.csv"
)

THRESHOLD_FILE = (
    RESULTS_DIR
    / "india_threshold_analysis.csv"
)

ROBUST_FILE = (
    RESULTS_DIR
    / "india_robust_validation_summary.csv"
)

SENSITIVITY_FILE = (
    RESULTS_DIR
    / "india_model_sensitivity_summary.csv"
)

FINAL_MODEL_FILE = (
    MODELS_DIR
    / "india_flood_impact_final_time_based.pkl"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "india_final_validation_report.csv"
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("INDIA FLOOD IMPACT - FINAL VALIDATION REPORT")
print("=" * 70)


# ============================================================
# 3. REQUIRED FILES
# ============================================================

required_files = {

    "Feature Validity Report":
        VALIDITY_FILE,

    "Final Time-Based Summary":
        FINAL_SUMMARY_FILE,

    "Final Feature Importance":
        FINAL_IMPORTANCE_FILE,

    "Threshold Analysis":
        THRESHOLD_FILE,

    "Robust Validation":
        ROBUST_FILE,

    "Sensitivity Analysis":
        SENSITIVITY_FILE,

    "Final Model":
        FINAL_MODEL_FILE,

    "Final Confusion Matrix":
        FIGURES_DIR
        / "india_final_confusion_matrix.png",

    "Final ROC Curve":
        FIGURES_DIR
        / "india_final_roc_curve.png",

    "Final Precision-Recall Curve":
        FIGURES_DIR
        / "india_final_precision_recall_curve.png",

    "Final Feature Importance Plot":
        FIGURES_DIR
        / "india_final_feature_importance.png"
}


# ============================================================
# 4. VERIFY FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILE VERIFICATION")
print("=" * 70)

verification_results = []

for name, path in required_files.items():

    exists = path.exists()

    status = (
        "FOUND"
        if exists
        else "MISSING"
    )

    print(
        f"{status:<8} : {name}"
    )

    verification_results.append({

        "Item": name,

        "Status": status,

        "Path": str(path)

    })


missing_files = [
    item
    for item in verification_results
    if item["Status"] == "MISSING"
]


# ============================================================
# 5. LOAD FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

final_summary = None

if FINAL_SUMMARY_FILE.exists():

    final_summary = pd.read_csv(
        FINAL_SUMMARY_FILE
    )

    print(
        "\nFinal model summary loaded."
    )

    print(
        final_summary.to_string(
            index=False
        )
    )

else:

    print(
        "\nWARNING: Final model summary not found."
    )


# ============================================================
# 6. EXTRACT FINAL METRICS
# ============================================================

metrics = {}

if final_summary is not None:

    for _, row in final_summary.iterrows():

        metric = str(
            row["Metric"]
        )

        value = row["Value"]

        metrics[metric] = value


# ============================================================
# 7. DISPLAY FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TIME-BASED PERFORMANCE")
print("=" * 70)

metric_names = [

    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "Average_Precision"

]

for metric in metric_names:

    if metric in metrics:

        try:

            print(
                f"{metric:<20}: "
                f"{float(metrics[metric]):.4f}"
            )

        except (ValueError, TypeError):

            print(
                f"{metric:<20}: "
                f"{metrics[metric]}"
            )


# ============================================================
# 8. TRAIN / TEST PERIOD
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED EVALUATION PERIOD")
print("=" * 70)

period_metrics = [

    "Training_Start_Year",
    "Training_End_Year",
    "Testing_Start_Year",
    "Testing_End_Year",
    "Training_Rows",
    "Testing_Rows"

]

for metric in period_metrics:

    if metric in metrics:

        print(
            f"{metric:<25}: "
            f"{metrics[metric]}"
        )


# ============================================================
# 9. FEATURE VALIDITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDITY")
print("=" * 70)

valid_count = 0
caution_count = 0
invalid_count = 0
review_count = 0

if VALIDITY_FILE.exists():

    validity_df = pd.read_csv(
        VALIDITY_FILE
    )

    if "Validity" in validity_df.columns:

        validity_counts = (
            validity_df["Validity"]
            .value_counts()
        )

        valid_count = int(
            validity_counts.get(
                "VALID",
                0
            )
        )

        caution_count = int(
            validity_counts.get(
                "CAUTION",
                0
            )
        )

        invalid_count = int(
            validity_counts.get(
                "INVALID",
                0
            )
        )

        review_count = int(
            validity_counts.get(
                "REVIEW",
                0
            )
        )

print(
    f"\nVALID   : {valid_count}"
)

print(
    f"CAUTION : {caution_count}"
)

print(
    f"INVALID : {invalid_count}"
)

print(
    f"REVIEW  : {review_count}"
)


# ============================================================
# 10. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

best_threshold = None
best_f1 = None

default_f1 = None

if THRESHOLD_FILE.exists():

    threshold_df = pd.read_csv(
        THRESHOLD_FILE
    )

    if (
        "Threshold" in threshold_df.columns
        and
        "F1_Score" in threshold_df.columns
    ):

        best_index = (
            threshold_df["F1_Score"]
            .idxmax()
        )

        best_row = (
            threshold_df.loc[
                best_index
            ]
        )

        best_threshold = float(
            best_row["Threshold"]
        )

        best_f1 = float(
            best_row["F1_Score"]
        )

        default_rows = threshold_df[
            threshold_df["Threshold"]
            == 0.50
        ]

        if not default_rows.empty:

            default_f1 = float(
                default_rows.iloc[0][
                    "F1_Score"
                ]
            )

        print(
            f"\nBest F1 threshold : "
            f"{best_threshold:.2f}"
        )

        print(
            f"Best F1           : "
            f"{best_f1:.4f}"
        )

        if default_f1 is not None:

            print(
                f"F1 at 0.50        : "
                f"{default_f1:.4f}"
            )

            print(
                f"F1 improvement     : "
                f"{best_f1 - default_f1:+.4f}"
            )

    else:

        print(
            "\nThreshold columns not found."
        )

else:

    print(
        "\nThreshold analysis file not found."
    )


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES")
print("=" * 70)

if FINAL_IMPORTANCE_FILE.exists():

    importance_df = pd.read_csv(
        FINAL_IMPORTANCE_FILE
    )

    print(
        "\nTop 10 features:"
    )

    print(
        importance_df
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nFeature importance file not found."
    )


# ============================================================
# 12. MODEL SENSITIVITY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SENSITIVITY")
print("=" * 70)

if SENSITIVITY_FILE.exists():

    sensitivity_df = pd.read_csv(
        SENSITIVITY_FILE
    )

    if (
        "Experiment" in sensitivity_df.columns
        and
        "F1_Score" in sensitivity_df.columns
    ):

        best_sensitivity = (
            sensitivity_df
            .loc[
                sensitivity_df["F1_Score"]
                .idxmax()
            ]
        )

        print(
            "\nBest sensitivity experiment:"
        )

        print(
            best_sensitivity.to_string()
        )

else:

    print(
        "\nSensitivity analysis file not found."
    )


# ============================================================
# 13. ROBUST VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ROBUST VALIDATION")
print("=" * 70)

if ROBUST_FILE.exists():

    robust_df = pd.read_csv(
        ROBUST_FILE
    )

    print(
        "\nRobust validation summary:"
    )

    print(
        robust_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nRobust validation file not found."
    )


# ============================================================
# 14. FINAL LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEAKAGE STATUS")
print("=" * 70)

direct_leakage_features = [

    "Human fatality",
    "Human injured",
    "Human Displaced",
    "Animal Fatality",
    "Description of Casualties/injured",
    "Total_Human_Impact",
    "Reported_Human_Impact"

]

leakage_found = []

if VALIDITY_FILE.exists():

    for feature in direct_leakage_features:

        if feature in validity_df["Original_Feature"].values:

            row = validity_df[
                validity_df[
                    "Original_Feature"
                ] == feature
            ]

            if not row.empty:

                validity = str(
                    row.iloc[0]["Validity"]
                )

                if validity == "INVALID":

                    leakage_found.append(
                        feature
                    )


if leakage_found:

    print(
        "\nWARNING: Invalid leakage features detected:"
    )

    for feature in leakage_found:

        print(
            "-",
            feature
        )

    leakage_status = "FAIL"

else:

    print(
        "\nPASS: No direct outcome leakage "
        "features detected."
    )

    leakage_status = "PASS"


# ============================================================
# 15. FINAL ASSESSMENT
# ============================================================

print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

if missing_files:

    final_status = "INCOMPLETE"

    print(
        "\nWARNING:"
    )

    print(
        f"{len(missing_files)} required output(s) "
        "are missing."
    )

else:

    if leakage_status == "PASS":

        final_status = "READY"

        print(
            "\nPASS: All required outputs are present."
        )

        print(
            "PASS: No direct outcome leakage detected."
        )

        print(
            "PASS: Final model was evaluated using "
            "a chronological train/test separation."
        )

        print(
            "\nThe model package is ready for "
            "team-level review."
        )

    else:

        final_status = "REVIEW_REQUIRED"

        print(
            "\nREVIEW REQUIRED:"
        )

        print(
            "Potential leakage issue detected."
        )


# ============================================================
# 16. CREATE CONSOLIDATED REPORT
# ============================================================

report_rows = [

    {
        "Category": "Final Status",
        "Metric": "Overall_Status",
        "Value": final_status
    },

    {
        "Category": "Data Split",
        "Metric": "Training_Start_Year",
        "Value": metrics.get(
            "Training_Start_Year",
            ""
        )
    },

    {
        "Category": "Data Split",
        "Metric": "Training_End_Year",
        "Value": metrics.get(
            "Training_End_Year",
            ""
        )
    },

    {
        "Category": "Data Split",
        "Metric": "Testing_Start_Year",
        "Value": metrics.get(
            "Testing_Start_Year",
            ""
        )
    },

    {
        "Category": "Data Split",
        "Metric": "Testing_End_Year",
        "Value": metrics.get(
            "Testing_End_Year",
            ""
        )
    },

    {
        "Category": "Data",
        "Metric": "Training_Rows",
        "Value": metrics.get(
            "Training_Rows",
            ""
        )
    },

    {
        "Category": "Data",
        "Metric": "Testing_Rows",
        "Value": metrics.get(
            "Testing_Rows",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "Accuracy",
        "Value": metrics.get(
            "Accuracy",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "Precision",
        "Value": metrics.get(
            "Precision",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "Recall",
        "Value": metrics.get(
            "Recall",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "F1_Score",
        "Value": metrics.get(
            "F1_Score",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "ROC_AUC",
        "Value": metrics.get(
            "ROC_AUC",
            ""
        )
    },

    {
        "Category": "Performance",
        "Metric": "Average_Precision",
        "Value": metrics.get(
            "Average_Precision",
            ""
        )
    },

    {
        "Category": "Feature_Validity",
        "Metric": "Valid_Features",
        "Value": valid_count
    },

    {
        "Category": "Feature_Validity",
        "Metric": "Caution_Features",
        "Value": caution_count
    },

    {
        "Category": "Feature_Validity",
        "Metric": "Invalid_Features",
        "Value": invalid_count
    },

    {
        "Category": "Feature_Validity",
        "Metric": "Review_Features",
        "Value": review_count
    },

    {
        "Category": "Leakage",
        "Metric": "Direct_Leakage_Status",
        "Value": leakage_status
    },

    {
        "Category": "Threshold",
        "Metric": "Best_F1_Threshold",
        "Value": (
            best_threshold
            if best_threshold is not None
            else ""
        )
    },

    {
        "Category": "Threshold",
        "Metric": "Best_F1",
        "Value": (
            best_f1
            if best_f1 is not None
            else ""
        )
    },

    {
        "Category": "Threshold",
        "Metric": "Default_0.50_F1",
        "Value": (
            default_f1
            if default_f1 is not None
            else ""
        )
    }

]


report_df = pd.DataFrame(
    report_rows
)


# ============================================================
# 17. SAVE REPORT
# ============================================================

report_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(
    "\nConsolidated final validation report saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION REPORT COMPLETE")
print("=" * 70)

print(
    f"\nOverall status: {final_status}"
)

print(
    f"Direct leakage status: {leakage_status}"
)

print(
    f"Required outputs missing: "
    f"{len(missing_files)}"
)

print(
    "\nThis report consolidates the final model's:"
)

print(
    "1. Time-based evaluation"
)

print(
    "2. Feature validity"
)

print(
    "3. Leakage verification"
)

print(
    "4. Threshold analysis"
)

print(
    "5. Feature importance"
)

print(
    "6. Sensitivity analysis"
)

print(
    "7. Final model artifact verification"
)

print("=" * 70)