
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/employee_attrition.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. REMOVE ROW WITH MISSING TARGET
# ============================================================

TARGET_COLUMN = "attrition_flag"

df = df.dropna(subset=[TARGET_COLUMN])


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN].astype(int)


# Remove employee ID because it is only an identifier
X = X.drop(columns=["employee_id"])


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

numerical_features = [
    "tenure_years",
    "performance_rating",
    "training_hours_annual",
    "avg_overtime_hrs_week",
    "job_satisfaction_score",
    "work_life_balance_score",
    "last_promotion_years_ago",
]

categorical_features = [
    "department",
    "role_level",
    "salary_band",
]


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 80% train+validation
# 20% test

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 60% train
# 20% validation
# 20% test

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)


# ============================================================
# 6. PREPROCESSING PIPELINES
# ============================================================

# Used for models where scaling is important:
# Logistic Regression, KNN, SVM

scaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# Used for tree-based models:
# Decision Tree, Random Forest, Gradient Boosting, XGBoost

unscaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            numerical_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": (
        scaled_preprocessor,
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),

    "KNN": (
        scaled_preprocessor,
        KNeighborsClassifier(
            n_neighbors=5,
        ),
    ),

    "Decision Tree": (
        unscaled_preprocessor,
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        ),
    ),

    "Random Forest": (
        unscaled_preprocessor,
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    ),

    "Gradient Boosting": (
        unscaled_preprocessor,
        GradientBoostingClassifier(
            random_state=42,
        ),
    ),

    "XGBoost": (
        unscaled_preprocessor,
        XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    ),

    "SVM": (
        scaled_preprocessor,
        SVC(
            probability=True,
            random_state=42,
        ),
    ),
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for model_name, (preprocessor, model) in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # Create complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # --------------------------------------------------------
    # Training time
    # --------------------------------------------------------

    start_train = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start_train

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    start_predict = time.perf_counter()

    train_predictions = pipeline.predict(X_train)
    val_predictions = pipeline.predict(X_val)

    prediction_time = time.perf_counter() - start_predict

    # --------------------------------------------------------
    # Probability predictions for ROC-AUC
    # --------------------------------------------------------

    train_probabilities = pipeline.predict_proba(X_train)[:, 1]
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    # --------------------------------------------------------
    # Training metrics
    # --------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    train_f1 = f1_score(
        y_train,
        train_predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_roc_auc = roc_auc_score(
        y_val,
        val_probabilities,
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        {
            "Model": model_name,
            "Train Accuracy": train_accuracy,
            "Train F1": train_f1,
            "Validation Accuracy": val_accuracy,
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1": val_f1,
            "Validation ROC-AUC": val_roc_auc,
            "Training Time (sec)": training_time,
            "Prediction Time (sec)": prediction_time,
        }
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Training Accuracy:    {train_accuracy:.4f}")
    print(f"Training F1:          {train_f1:.4f}")
    print(f"Validation Accuracy:  {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall:    {val_recall:.4f}")
    print(f"Validation F1:        {val_f1:.4f}")
    print(f"Validation ROC-AUC:   {val_roc_auc:.4f}")
    print(f"Training Time:        {training_time:.4f} sec")
    print(f"Prediction Time:      {prediction_time:.4f} sec")


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 10. BEST MODEL BASED ON VALIDATION F1
# ============================================================

best_model = results_df.loc[
    results_df["Validation F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST VALIDATION MODEL")
print("=" * 70)

print("Model:", best_model["Model"])
print(
    "Validation F1:",
    f"{best_model['Validation F1']:.4f}",
)
print(
    "Validation ROC-AUC:",
    f"{best_model['Validation ROC-AUC']:.4f}",
)





# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

best_model_name = best_model["Model"]

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print("Selected model:", best_model_name)


# Get the selected model and its preprocessing
selected_preprocessor, selected_classifier = models[best_model_name]


# Create a fresh pipeline
final_pipeline = Pipeline(
    steps=[
        ("preprocessor", selected_preprocessor),
        ("model", selected_classifier),
    ]
)


# ------------------------------------------------------------
# Train selected model on training + validation data
# ------------------------------------------------------------

X_final_train = pd.concat([X_train, X_val])
y_final_train = pd.concat([y_train, y_val])


final_pipeline.fit(
    X_final_train,
    y_final_train,
)


# ------------------------------------------------------------
# Test predictions
# ------------------------------------------------------------

test_predictions = final_pipeline.predict(X_test)

test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# Test metrics
# ------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)


# ------------------------------------------------------------
# Print test results
# ------------------------------------------------------------

print(f"Test Accuracy:    {test_accuracy:.4f}")
print(f"Test Precision:   {test_precision:.4f}")
print(f"Test Recall:      {test_recall:.4f}")
print(f"Test F1:          {test_f1:.4f}")
print(f"Test ROC-AUC:     {test_roc_auc:.4f}")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    test_predictions,
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nMatrix format:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")


# ============================================================
# 13. VALIDATION VS TEST
# ============================================================

validation_f1 = best_model["Validation F1"]
validation_roc_auc = best_model["Validation ROC-AUC"]

print("\n" + "=" * 70)
print("VALIDATION VS TEST")
print("=" * 70)

print(f"Validation F1:    {validation_f1:.4f}")
print(f"Test F1:          {test_f1:.4f}")

print(f"\nValidation ROC-AUC: {validation_roc_auc:.4f}")
print(f"Test ROC-AUC:       {test_roc_auc:.4f}")


f1_difference = test_f1 - validation_f1

print(f"\nF1 difference: {f1_difference:+.4f}")
