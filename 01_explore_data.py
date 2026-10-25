
import pandas as pd

# Step 1: Load the customer churn dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Step 2: Display the first 5 customers
print("\n--- First 5 Customers ---")
print(df.head())

# Step 3: Check the number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# Step 4: Display all column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Step 5: Check data types and missing values
print("\n--- Dataset Information ---")
df.info()

# Step 6: Count customers who stayed and left
print("\n--- Churn Distribution ---")
print(df["Churn"].value_counts())
