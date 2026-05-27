import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set up page configurations
st.set_page_config(page_title="Student Placement Predictor", layout="centered")
st.title("🎓 Student Placement Predictor")
st.write("Enter the student details below to check their placement probability.")

# Load the saved model artifacts
@st.cache_resource
def load_artifacts():
    with open("placement_model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

# Create layout columns for the inputs
col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"])
    college_tier = st.selectbox("College Tier", [1, 2, 3], index=1)
    coding_score = st.number_input("Coding Score", min_value=0.0, max_value=100.0, value=50.0)
    communication_score = st.number_input("Communication Score", min_value=0.0, max_value=10.0, value=7.0)
    aptitude_score = st.number_input("Aptitude Score", min_value=0.0, max_value=100.0, value=65.0)
    resume_score = st.number_input("Resume Score", min_value=0.0, max_value=100.0, value=70.0)

with col2:
    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
    projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=2)
    backlogs = st.number_input("Active Backlogs", min_value=0, max_value=10, value=0)
    skill_score = st.slider("Overall Skill Score", min_value=1, max_value=5, value=3)
    
    st.write("**Core Skills Technical Checkboxes:**")
    python_skill = 1 if st.checkbox("Python Skill", value=True) else 0
    dsa_skill = 1 if st.checkbox("DSA Skill", value=True) else 0
    ml_skill = 1 if st.checkbox("Machine Learning Skill") else 0
    web_dev_skill = 1 if st.checkbox("Web Development Skill") else 0

# Predict Button
if st.button("Predict Placement Status", type="primary"):
    # 1. Structure the raw inputs as a DataFrame matching the original training format
    raw_input_df = pd.DataFrame([{
        'cgpa': cgpa,
        'branch': branch,
        'college_tier': college_tier,
        'python_skill': python_skill,
        'dsa_skill': dsa_skill,
        'ml_skill': ml_skill,
        'web_dev_skill': web_dev_skill,
        'coding_score': coding_score,
        'communication_score': communication_score,
        'aptitude_score': aptitude_score,
        'internships': internships,
        'projects': projects,
        'backlogs': backlogs,
        'resume_score': resume_score,
        'skill_score': skill_score
    }])
    
    # 2. One-hot encode the categorical features
    encoded_features = encoder.transform(raw_input_df[['branch']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['branch']), index=raw_input_df.index)
    
    # 3. Reconstruct full encoded dataframe in exact order
    numerical_df = raw_input_df.drop(columns=['branch'])
    full_encoded_df = pd.concat([numerical_df, encoded_df], axis=1)
    
    # 4. Scale inputs using the saved scaler
    scaled_input = scaler.transform(full_encoded_df)
    
    # 5. Make the prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0] # Grab probabilities if supported (Logistic Regression/SVM)
    
    # Display Results
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **Placed!** Probability of placement: {probabilities[1]*100:.2f}%")
    else:
        st.error(f"⚠️ **Not Placed Yet.** Probability of placement: {probabilities[1]*100:.2f}%")