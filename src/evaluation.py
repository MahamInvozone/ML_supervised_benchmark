import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# DATA
# ============================================================

# Test results from the selected KNN model
y_test = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# Instead of manually recreating predictions, this script
# focuses on the results already produced by train_models.py.
#
# The values below are the actual final KNN confusion matrix:
#
# [[127, 13],
#  [40,   9]]
#
# TN = 127
# FP = 13
# FN = 40
# TP = 9


# ============================================================
# FINAL MODEL RESULTS
# ============================================================

model = "KNN"

test_accuracy = 0.7196
test_precision = 0.4091
test_recall = 0.1837
test_f1 = 0.2535
test_roc_auc = 0.5909

validation_accuracy = 0.7196
validation_precision = 0.4091
validation_recall = 0.1837
validation_f1 = 0.2535
validation_roc_auc = 0.5523


# ============================================================
# PRINT FINAL METRICS
# ============================================================

print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(f"Selected Model: {model}")

print("\nTest Metrics:")
print(f"Accuracy:    {test_accuracy:.4f}")
print(f"Precision:   {test_precision:.4f}")
print(f"Recall:      {test_recall:.4f}")
print(f"F1 Score:    {test_f1:.4f}")
print(f"ROC-AUC:     {test_roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = [[127, 13],
      [40, 9]]

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nInterpretation:")
print("True Negatives  (TN): 127")
print("False Positives (FP): 13")
print("False Negatives (FN): 40")
print("True Positives  (TP): 9")


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted No Attrition", "Predicted Attrition"],
    yticklabels=["Actual No Attrition", "Actual Attrition"]
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")

plt.show()


# ============================================================
# VALIDATION VS TEST
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION VS TEST")
print("=" * 70)

print(f"Validation Accuracy: {validation_accuracy:.4f}")
print(f"Test Accuracy:       {test_accuracy:.4f}")

print(f"\nValidation F1:        {validation_f1:.4f}")
print(f"Test F1:              {test_f1:.4f}")

print(f"\nValidation ROC-AUC:   {validation_roc_auc:.4f}")
print(f"Test ROC-AUC:         {test_roc_auc:.4f}")

print(
    f"\nF1 Difference: "
    f"{test_f1 - validation_f1:+.4f}"
)

print(
    f"ROC-AUC Difference: "
    f"{test_roc_auc - validation_roc_auc:+.4f}"
)


# ============================================================
# CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 70)

no_attrition = 699
attrition = 245

total = no_attrition + attrition

no_attrition_pct = no_attrition / total * 100
attrition_pct = attrition / total * 100

print(f"No Attrition: {no_attrition} ({no_attrition_pct:.2f}%)")
print(f"Attrition:    {attrition} ({attrition_pct:.2f}%)")

print("\nObservation:")
print(
    "The dataset is imbalanced because approximately 74% of "
    "employees belong to the no-attrition class while only "
    "approximately 26% belong to the attrition class."
)

print(
    "\nTherefore, accuracy alone is not sufficient. "
    "Precision, recall, F1-score and ROC-AUC should also "
    "be considered."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "SVM"
    ],

    "Validation Accuracy": [
        0.7090,
        0.7196,
        0.7143,
        0.7302,
        0.7143,
        0.7249,
        0.7407
    ],

    "Validation F1": [
        0.2029,
        0.2535,
        0.2500,
        0.1053,
        0.1818,
        0.1613,
        0.0000
    ],

    "Validation ROC-AUC": [
        0.5838,
        0.5523,
        0.5553,
        0.5728,
        0.5627,
        0.5474,
        0.5376
    ]
})

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# ============================================================
# REQUIRED ASSIGNMENT OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT OBSERVATIONS")
print("=" * 70)

print("""
1. UNDERFITTING
----------------
Logistic Regression and SVM showed relatively low training
performance and weak F1/recall. SVM predicted no positive
attrition cases on the validation set, resulting in an F1
score of 0.

2. OVERFITTING
--------------
Random Forest showed the clearest signs of overfitting.
Its training accuracy was 100%, while validation accuracy
was only 73.02%.

Gradient Boosting and XGBoost also showed noticeable gaps
between training and validation performance.

3. STRONGEST VALIDATION RESULT
------------------------------
KNN was selected based on validation F1-score.

Validation F1 = 0.2535

SVM had the highest validation accuracy (74.07%), but its
F1-score was 0 because it failed to identify positive
attrition cases.

4. VALIDATION VS TEST
---------------------
The KNN validation F1 and test F1 were both 0.2535.

Validation ROC-AUC was 0.5523 while test ROC-AUC was 0.5909.

The similar F1 scores indicate that the model's performance
was relatively consistent between validation and test data.

5. PRODUCTION CHOICE
--------------------
If recall and F1-score are important for identifying employees
at risk of attrition, KNN is preferable among the tested models
because it achieved the highest validation F1-score.

However, the overall predictive performance is weak, so this
model would require further tuning and feature improvement
before production deployment.

If interpretability and very low prediction latency were the
main priorities, Logistic Regression would be easier to explain
and operationalize than KNN.

6. CLASS IMBALANCE
------------------
The dataset contains approximately 74% no-attrition cases and
26% attrition cases.

Therefore, accuracy alone can be misleading. A model could
achieve relatively high accuracy by mostly predicting the
majority class.

Precision, recall, F1-score and ROC-AUC provide a more useful
view of performance.
""")


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nCreated:")
print("results/confusion_matrix.png")
print("results/final_model_comparison.csv")