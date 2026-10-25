
import pandas as pd

# Step 1: Load the cleaned dataset
df = pd.read_csv("data/cleaned_churn.csv")

# Step 2: Separate input features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Step 3: Convert categorical columns into numerical columns
X = pd.get_dummies(X, dtype=int)

# Step 4: Display the results
print("--- Encoded Features ---")
print("Shape of X:", X.shape)

print("\nFirst five encoded rows:")
print(X.head())

print("\n--- Target Variable ---")
print(y.value_counts())

# Step 5: Save the prepared data
X.to_csv("data/encoded_features.csv", index=False)
y.to_csv("data/target.csv", index=False)

print("\nEncoded features and target saved successfully!")
