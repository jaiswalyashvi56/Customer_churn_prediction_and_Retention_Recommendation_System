
import pandas as pd

# Step 1: Load the dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Step 2: Display the first five customers
print("\n--- First Five Customer Records ---")
print(df.head())

# Step 3: Display dataset size
print("\n--- Dataset Shape ---")
print(df.shape)

# Step 4: Display column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Step 5: Display data types
print("\n--- Data Types ---")
print(df.dtypes)

# Step 6: Display target distribution
print("\n--- Churn Distribution ---")
print(df["Churn"].value_counts())
