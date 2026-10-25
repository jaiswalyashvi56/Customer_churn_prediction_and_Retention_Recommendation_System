
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE_DIR = Path(__file__).resolve().parent


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction & Retention Recommendation System")
st.write(
    "Estimate customer churn risk using an Artificial Neural Network "
    "and view explainable retention suggestions."
)

def get_retention_recommendation(
    contract,
    tenure,
    monthly_charges,
    internet_service,
    tech_support
):
    recommendations = []

    # Rule 1: Month-to-month contract
    if contract == "Month-to-month":
        recommendations.append(
            "Offer a suitable long-term contract with "
            "clear pricing and flexible terms."
        )

    # Rule 2: New customer
    if tenure < 12:
        recommendations.append(
            "Provide an onboarding check-in and "
            "help the customer understand available services."
        )

    # Rule 3: High monthly charges
    if monthly_charges >= 80:
        recommendations.append(
            "Review the customer's bill and check "
            "whether a suitable lower-cost plan is available."
        )

    # Rule 4: No technical support
    if (
        internet_service != "No"
        and tech_support == "No"
    ):
        recommendations.append(
            "Explain available technical support options "
            "and check whether the customer needs assistance."
        )

    # Default recommendation
    if not recommendations:
        recommendations.append(
            "Offer a routine satisfaction check-in "
            "and ask about any service concerns."
        )

    return recommendations


# Run the test only when this file is executed directly
if __name__ == "__main__":
    example_recommendations = get_retention_recommendation(
        contract="Month-to-month",
        tenure=3,
        monthly_charges=95,
        internet_service="Fiber optic",
        tech_support="No"
    )

    print("--- Retention Recommendations ---")

    for number, recommendation in enumerate(
        example_recommendations, start=1
    ):
        print(f"{number}. {recommendation}")


# -------------------------------
# Load saved model and files
# -------------------------------


@st.cache_resource
def load_model_and_preprocessing():
    model = tf.keras.models.load_model(
        BASE_DIR / "churn_ann.keras"
    )
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_columns = joblib.load(
        BASE_DIR / "feature_columns.pkl"
    )
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model_and_preprocessing()

# Load recommendation function from 07_recommendation.py
recommendation_path = BASE_DIR / "07_recommendation.py"
spec = spec_from_file_location(
    "retention_rules", recommendation_path
)
retention_rules = module_from_spec(spec)
spec.loader.exec_module(retention_rules)

get_retention_recommendation = (
    retention_rules.get_retention_recommendation
)

# -------------------------------
# Customer input form
# -------------------------------
st.header("Enter Customer Details")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.number_input(
            "Tenure (months)", min_value=0,
            max_value=72, value=12
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn"
    )

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer])

    # Encode the input in the same way as training
    input_df = pd.get_dummies(
        input_df, dtype=int
    )
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the saved scaler
    input_scaled = scaler.transform(input_df)

    # Predict churn probability
    churn_probability = float(
        model.predict(input_scaled, verbose=0)[0][0]
    )

    st.subheader("Prediction Result")
    st.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.1%}"
    )
    st.progress(churn_probability)

    
    if churn_probability >= 0.70:
        st.error("High Churn Risk")
    elif churn_probability >= 0.40:
        st.warning("Medium Churn Risk")
    else:
        st.success("Low Churn Risk")


    st.caption(
        "This is a model estimate, not a guarantee of customer behavior. "
        "The 0.5 threshold is an initial choice and can be tuned."
    )

    # -------------------------------
    # Retention recommendations
    # -------------------------------
    st.subheader("Suggested Retention Actions")

    recommendations = get_retention_recommendation(
        contract=contract,
        tenure=tenure,
        monthly_charges=monthly_charges,
        internet_service=internet_service,
        tech_support=tech_support
    )

    for item in recommendations:
        st.markdown(f"- {item}")

    
# -------------------------------
# Customer Churn Dashboard
# -------------------------------

st.divider()
st.header("📈 Customer Churn Overview")

dashboard_df = pd.read_csv(
    BASE_DIR / "data" / "cleaned_churn.csv"
)

total_customers = len(dashboard_df)

churned_customers = (
    dashboard_df["Churn"].astype(str).str.strip().str.lower() == "yes"
).sum()

retained_customers = (
    dashboard_df["Churn"].astype(str).str.strip().str.lower() == "no"
).sum()

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Churn Rate", f"{churn_rate:.1f}%")

st.subheader("Customer Churn Distribution")

chart_data = pd.DataFrame({
    "Customer Status": ["Stayed", "Churned"],
    "Number of Customers": [
        int(retained_customers),
        int(churned_customers)
    ]
})

st.bar_chart(
    chart_data,
    x="Customer Status",
    y="Number of Customers",
    height=350
)

