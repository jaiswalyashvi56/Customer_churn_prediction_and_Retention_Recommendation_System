
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the original dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Step 2: Clean TotalCharges for numerical analysis
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

# Step 3: Calculate churn percentages
print("--- Churn Percentage ---")
print(
    (df["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)

# Step 4: Create a 2x2 chart layout
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Overall churn distribution
sns.countplot(data=df, x="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Customer Churn Distribution")
axes[0, 0].set_xlabel("Churn")
axes[0, 0].set_ylabel("Number of Customers")

# Chart 2: Churn by contract
sns.countplot(
    data=df, x="Contract", hue="Churn", ax=axes[0, 1]
)
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].tick_params(axis="x", rotation=15)

# Chart 3: Monthly charges by churn
sns.boxplot(
    data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0]
)
axes[1, 0].set_title("Monthly Charges by Churn")

# Chart 4: Tenure by churn
sns.boxplot(
    data=df, x="Churn", y="tenure", ax=axes[1, 1]
)
axes[1, 1].set_title("Customer Tenure by Churn")

plt.tight_layout()
plt.show()
