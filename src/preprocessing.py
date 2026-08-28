
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/employee_attrition.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. REMOVE ROWS WITH MISSING TARGET
# ============================================================

TARGET_COLUMN = "attrition_flag"

df = df.dropna(subset=[TARGET_COLUMN])


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# Remove employee ID because it is only an identifier
X = X.drop(columns=["employee_id"])


# ============================================================
# 4. IDENTIFY FEATURE TYPES
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
# 5. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ]
)


# ============================================================
# 7. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# 8. TEST PREPROCESSOR
# ============================================================

X_transformed = preprocessor.fit_transform(X)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)

print("Original feature shape:", X.shape)

print(
    "Transformed feature shape:",
    X_transformed.shape
)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nPreprocessing steps:")

print("\nNumerical:")
print("- Missing values → median")
print("- Scaling → StandardScaler")

print("\nCategorical:")
print("- Missing values → most frequent")
print("- Encoding → OneHotEncoder")

print("\nPreprocessing completed successfully.")