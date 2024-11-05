import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from preprocessing_steps import CleanAndPreprocess, DynamicEncodeVariables, FeatureEngineering
import joblib

# Load model and preprocessing pipeline
best_rf_model = joblib.load("best_rf_model.pkl")
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")

# Function to plot the probability bar with a pointer
def plot_probability_bar(probability):
    fig, ax = plt.subplots(figsize=(8, 1))

    # Draw the probability bar
    ax.barh([0], [0.5], color='red', edgecolor='black')  # Red part (0% - 50%)
    ax.barh([0], [0.5], left=[0.5], color='green', edgecolor='black')  # Green part (50% - 100%)

    # Set limits and remove ticks
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate with labels
    ax.text(0.25, 0.5, "Loan Not Approved", ha='center', va='center', color='white')
    ax.text(0.75, 0.5, "Loan Approved", ha='center', va='center', color='white')

    # Place an arrow based on the probability
    ax.annotate('', xy=(probability, 0), xytext=(probability, 0.6),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=3.5, headwidth=8))

    st.pyplot(fig)

# Streamlit UI elements
st.title("Loan Eligibility Prediction")

# User inputs for loan features
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income per month in US$", value=0)
CoapplicantIncome = st.number_input("Coapplicant Income per month in US$", value=0)
LoanAmount = st.number_input("Loan Amount in thousand US$", value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (Loan repayment Period in months) ", value=360)
Credit_History = st.selectbox("Credit History(0-bad, 1 good)", [1.0, 0.0])
Property_Area = st.selectbox("Property Area (The location Of property)", ["Urban", "Semiurban", "Rural"])

# Determine Is_Rural based on Property_Area and remove Property_Area from input
Is_Rural = 1 if Property_Area == "Rural" else 0

# Create DataFrame from user input
user_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Is_Rural': [Is_Rural]  # Include Is_Rural instead of Property_Area
})

# Preprocess the input data
processed_data = preprocessing_pipeline.transform(user_data)

# Try to get feature names from the pipeline if possible
try:
    feature_names = preprocessing_pipeline[:-1].get_feature_names_out()
except AttributeError:
    feature_names = best_rf_model.feature_names_in_

# Ensure the processed data matches the feature names
processed_data_df = pd.DataFrame(processed_data, columns=feature_names)
processed_data_df = processed_data_df[feature_names]  # Reorder columns to match model training

# Predict the loan approval probability
predicted_proba = best_rf_model.predict_proba(processed_data_df)[0, 1]  # Probability of loan approval

# Display the probability bar
plot_probability_bar(predicted_proba)

# Show prediction result
prediction = "Approved" if predicted_proba >= 0.5 else "Not Approved"
st.write(f"Predicted Loan Status: {prediction}")

# Initialize the SHAP TreeExplainer with the trained model
explainer = shap.TreeExplainer(best_rf_model, feature_perturbation="interventional")

# Calculate SHAP values for the single instance
shap_values_single = explainer.shap_values(processed_data_df)

# Diagnostic print statements
print("explainer.expected_value:", explainer.expected_value)
print("Type of explainer.expected_value:", type(explainer.expected_value))
print("shap_values_single type:", type(shap_values_single))
print("shap_values_single shape (if array):", [arr.shape for arr in shap_values_single] if isinstance(shap_values_single, list) else shap_values_single.shape)

# Display SHAP waterfall plot for feature contribution
st.write("Feature Contribution to Prediction")
shap_fig = plt.figure(figsize=(10, 6))

# Explicitly get base value and SHAP values for class 1
base_value_class_1 = explainer.expected_value[1]  # Use only the second element for class 1
shap_values_for_class_1 = shap_values_single[0, :, 1]  # Use SHAP values for class 1

# Additional diagnostic print to ensure correct values
print("base_value_class_1:", base_value_class_1)
print("shap_values_for_class_1 shape:", shap_values_for_class_1.shape)

# Create the waterfall plot for class 1
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_for_class_1,        # SHAP values for class 1 (loan approval)
        base_values=base_value_class_1,        # Single base value for class 1
        data=processed_data_df.iloc[0].values, # Input data for the instance
        feature_names=processed_data_df.columns
    )
)
st.pyplot(shap_fig)


