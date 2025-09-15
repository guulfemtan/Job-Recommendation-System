import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# Load model and preprocessors
model = load_model("job_recommendation_model.keras")

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("job_encoder.pkl", "rb") as f:
    job_encoder = pickle.load(f)

st.title("Job Recommendation System")
st.write("Enter your skills to get job recommendations:")

# User input
skills_input = st.text_area("List your skills separated by commas:", "")

if st.button("Recommend Job"):
    if skills_input.strip() == "":
        st.warning("Please enter your skills first!")
    else:
        # Transform input
        skills_vector = vectorizer.transform([skills_input]).toarray()
        prediction = model.predict(skills_vector)
        recommended_job = job_encoder.inverse_transform([np.argmax(prediction)])
        st.success(f"Recommended Job: {recommended_job[0]}")
