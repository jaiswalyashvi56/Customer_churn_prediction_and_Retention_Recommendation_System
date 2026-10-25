
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
