# 1. Dataset Overview
import pandas as pd

df = pd.read_csv("bank-full.csv", sep=";")

print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())


# 2. Target Variable Distribution
print("Target variable counts:")
print(df["y"].value_counts())

print("\nTarget variable percentages:")
print(df["y"].value_counts(normalize=True) * 100)


# 3. Missing Values
print("Missing values:")
print(df.isnull().sum())


# 4. Unknown Values
print("Unknown values by categorical feature:")

for column in ["poutcome", "contact", "education", "job"]:
    print(f"\n{column}:")
    print(df[column].value_counts().get("unknown", 0))


# 5. Duplicate Records
print("Number of duplicate records:", df.duplicated().sum())


# 6. Numerical Summary
print("Numerical summary:")
print(df[["age", "balance", "campaign", "duration", "pdays", "previous"]].describe())


# 7. Skewness Analysis
print("Skewness of selected numerical features:")
print(df[["age", "balance", "campaign", "duration", "pdays", "previous"]].skew())


# 8. Correlation Analysis
numerical_features = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]

correlation_matrix = df[numerical_features].corr()

print("Correlation matrix:")
print(correlation_matrix)


# 9. Target Variable Distribution
target_counts = df["y"].value_counts()
target_percentages = df["y"].value_counts(normalize=True) * 100

print("Target variable counts:")
print(target_counts)

print("\nTarget variable percentages:")
print(target_percentages.round(2))


# 10. Missing and Unknown Values
print("Missing values:")
print(df.isnull().sum())

print("\nUnknown values in categorical features:")

categorical_features = [
    "job", "marital", "education",
    "default", "housing", "loan",
    "contact", "month", "poutcome", "y"
]

for column in categorical_features:
    unknown_count = (df[column] == "unknown").sum()
    if unknown_count > 0:
        percentage = (unknown_count / len(df)) * 100
        print(f"{column}: {unknown_count} ({percentage:.2f}%)")


# 11. Duplicate Records
duplicate_count = df.duplicated().sum()

print("Number of duplicate records:")
print(duplicate_count)


# 12. Numerical Summary and Skewness
numerical_features = [
    "age", "balance", "campaign",
    "duration", "pdays", "previous"
]

summary = df[numerical_features].agg(
    ["min", "max", "median", "skew"]
).T

print("Numerical summary:")
print(summary)


# 13. IQR Outlier Detection
numerical_features = [
    "balance", "campaign", "duration",
    "pdays", "previous"
]

for column in numerical_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"{column}:")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Potential outliers: {len(outliers)}")
    print()


# 14. Data Types and Dataset Structure
print("Dataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)


