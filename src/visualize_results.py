import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs("results", exist_ok=True)

# Model results from train_models.py
results = {
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "SVM"
    ],
    "Train Accuracy": [
        0.7562, 0.7933, 0.8304, 1.0000, 0.9258, 0.9629, 0.7650
    ],
    "Validation Accuracy": [
        0.7090, 0.7196, 0.7143, 0.7302, 0.7143, 0.7249, 0.7407
    ],
    "Validation Precision": [
        0.3500, 0.4091, 0.3913, 0.3750, 0.3529, 0.3846, 0.0000
    ],
    "Validation Recall": [
        0.1429, 0.1837, 0.1837, 0.0612, 0.1224, 0.1020, 0.0000
    ],
    "Validation F1": [
        0.2029, 0.2535, 0.2500, 0.1053, 0.1818, 0.1613, 0.0000
    ],
    "Validation ROC-AUC": [
        0.5838, 0.5523, 0.5553, 0.5728, 0.5627, 0.5474, 0.5376
    ]
}

df = pd.DataFrame(results)

print("=" * 60)
print("MODEL VISUALIZATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Validation Accuracy
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Validation Accuracy",
    y="Model"
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Validation Accuracy")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig("results/validation_accuracy.png")
plt.show()

# ---------------------------------------------------------
# 2. Validation F1 Score
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Validation F1",
    y="Model"
)

plt.title("Validation F1 Score Comparison")
plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig("results/validation_f1.png")
plt.show()

# ---------------------------------------------------------
# 3. ROC-AUC Comparison
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Validation ROC-AUC",
    y="Model"
)

plt.title("Validation ROC-AUC Comparison")
plt.xlabel("ROC-AUC")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig("results/validation_roc_auc.png")
plt.show()

# ---------------------------------------------------------
# 4. Training vs Validation Accuracy
# ---------------------------------------------------------

accuracy_df = df[
    ["Model", "Train Accuracy", "Validation Accuracy"]
].melt(
    id_vars="Model",
    var_name="Dataset",
    value_name="Accuracy"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=accuracy_df,
    x="Model",
    y="Accuracy",
    hue="Dataset"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/train_vs_validation_accuracy.png")
plt.show()

# ---------------------------------------------------------
# 5. Training vs Validation F1
# ---------------------------------------------------------

f1_df = df[
    ["Model", "Validation F1"]
].copy()

# Add training F1 from your output
f1_df["Train F1"] = [
    0.2581,
    0.4800,
    0.5826,
    1.0000,
    0.8333,
    0.9231,
    0.1739
]

f1_df = f1_df.melt(
    id_vars="Model",
    var_name="Dataset",
    value_name="F1"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=f1_df,
    x="Model",
    y="F1",
    hue="Dataset"
)

plt.title("Training vs Validation F1 Score")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/train_vs_validation_f1.png")
plt.show()

# ---------------------------------------------------------
# Save results table
# ---------------------------------------------------------

df.to_csv("results/model_comparison.csv", index=False)

print("\nResults saved to:")
print("results/model_comparison.csv")
print("results/validation_accuracy.png")
print("results/validation_f1.png")
print("results/validation_roc_auc.png")
print("results/train_vs_validation_accuracy.png")
print("results/train_vs_validation_f1.png")

print("\nVisualization completed successfully.")