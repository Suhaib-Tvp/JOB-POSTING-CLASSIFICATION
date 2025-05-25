# main.py

from app import (
    scrape_karkidi_jobs,
    assign_clusters_to_jobs,
    persist_models,
    load_models,
    predict_job_cluster
)

if __name__ == "__main__":
    # 1. Scrape jobs (set pages=2 for quick test, increase as needed)
    df_jobs = scrape_karkidi_jobs(pages=2)
    print("Sample scraped jobs:")
    print(df_jobs.head())

    # 2. Cluster jobs and save models
    clustered_jobs, model_used, tfidf_used = assign_clusters_to_jobs(df_jobs)
    print("Clustered jobs sample:")
    print(clustered_jobs.head())
    persist_models(model_used, tfidf_used)

    # 3. Predict cluster for a new skills input
    model, vectorizer = load_models()
    sample_skills = "AWS, Python, Data Science, Machine Learning"
    cluster = predict_job_cluster(sample_skills, model, vectorizer)
    print(f"Predicted cluster for '{sample_skills}': {cluster}")
