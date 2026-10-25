

import pandas as pd

# Load original data
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Convert TotalCharges from text to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values before cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Check the cleaned data
print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")
