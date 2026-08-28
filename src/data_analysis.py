import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/employee_attrition.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for column in df.columns:
    print(column)


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nColumns with missing values:")

missing_columns = missing_values[missing_values > 0]

if len(missing_columns) == 0:
    print("No missing values found.")
else:
    print(missing_columns)


# ============================================================
# 6. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# 7. NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

numerical_features = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

for column in numerical_features:
    print(column)


# ============================================================
# 8. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

categorical_features = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

for column in categorical_features:
    print(column)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

TARGET_COLUMN = "attrition_flag"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df[TARGET_COLUMN].value_counts(dropna=False))


# ============================================================
# 10. TARGET PERCENTAGES
# ============================================================

print("\n" + "=" * 60)
print("TARGET PERCENTAGES")
print("=" * 60)

print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)


# ============================================================
# 11. UNIQUE VALUES IN CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL VALUE COUNTS")
print("=" * 60)

for column in categorical_features:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))


# ============================================================
# 12. NUMERICAL FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 60)

print(df[numerical_features].describe())


# ============================================================
# 13. UNIQUE VALUES PER COLUMN
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE VALUES PER COLUMN")
print("=" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# ============================================================
# 14. CONSTANT COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("CONSTANT COLUMNS")
print("=" * 60)

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

if constant_columns:
    for column in constant_columns:
        print(column)
else:
    print("No constant columns found.")


# ============================================================
# 15. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ============================================================
# 16. LAST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)

print(df.tail())


# ============================================================
# 17. FINAL ANALYSIS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Target column: {TARGET_COLUMN}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Duplicate rows: {duplicate_count}")
print(f"Missing-value columns: {len(missing_columns)}")

print("\nData analysis completed successfully.")
