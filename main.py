# main.py

import streamlit as st
import pandas as pd
from app import (
    scrape_karkidi_jobs,
    train_and_save_model,
    load_models,
    predict_job_cluster
)

st.title("Job Posting Cluster Predictor")
st.write("Enter your skills (comma separated) to predict the job cluster.")

# Try to load models
model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.info("Training model for the first time. This may take a minute...")
    df = scrape_karkidi_jobs(pages=1)
    if df.empty:
        st.error("Failed to scrape job data. Please try again later.")
        st.stop()
    model, vectorizer = train_and_save_model(df)
    st.success("Model trained and saved!")

skills = st.text_input("Enter skills (e.g., Python, SQL, Machine Learning):")

if skills:
    try:
        cluster = predict_job_cluster(skills, model, vectorizer)
        st.success(f"Predicted Cluster: {cluster}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Demo app. Clustering is based on scraped job skills and may be approximate.")
