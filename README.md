# Customer Churn Prediction and Retention Recommendation System Using ANN

## Project Overview
This project predicts whether a telecom customer is likely to leave a service using an Artificial Neural Network (ANN). It also provides retention recommendations to help businesses identify at-risk customers and take preventive action.

The project includes a machine learning workflow and an interactive Streamlit dashboard.

## Objectives
- Predict customer churn using an ANN model.
- Estimate customer churn probability and risk level.
- Explore customer data and identify churn patterns.
- Generate rule-based retention recommendations.
- Display results through an interactive dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## Project Workflow
1. Data exploration and analysis
2. Data cleaning and preprocessing
3. Encoding categorical variables
4. Train-test splitting and feature scaling
5. ANN model training and evaluation
6. Customer churn prediction
7. Retention recommendation generation
8. Streamlit dashboard visualization

## Features
- Interactive customer churn prediction
- Churn probability and risk indication
- Retention recommendations
- Data visualizations
- ANN-based classification

## Project Structure
- `app.py` - Streamlit application
- `01_explore_data.py` - Initial data exploration
- `02_eda.py` - Exploratory data analysis
- `03_preprocessing.py` - Data preprocessing
- `04_prepare_data.py` - Feature preparation
- `05_split_scale.py` - Data splitting and scaling
- `06_train_ann.py` - ANN model training
- `07_recommendation.py` - Retention recommendation logic
- `churn_ann.keras` - Trained ANN model
- `scaler.pkl` - Saved feature scaler
- `feature_columns.pkl` - Saved feature column information
- `requirements.txt` - Python dependencies

## Installation and Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/customer-churn-prediction-ann.git
   ```

2. Move into the project folder:

   ```bash
   cd customer-churn-prediction-ann
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

## Dataset
The project uses a telecom customer churn dataset. Refer to the dataset's original source and usage terms before redistributing it. Update this section with the actual dataset source used in the project.

## Model Evaluation
The ANN model is evaluated using classification metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

Add your final verified test results here.

## Screenshots
Add screenshots of the Streamlit dashboard and prediction results in this section.

## Future Improvements
- Compare ANN with other classification algorithms.
- Improve model interpretability.
- Evaluate retention strategies using business metrics.
- Deploy the application online.

## Author
Yashvi Jaiswal

## Disclaimer
This project is developed for educational and portfolio purposes.
Predictions are estimates and should be evaluated before being used for business decisions.
