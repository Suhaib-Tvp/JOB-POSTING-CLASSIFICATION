# app.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def scrape_karkidi_jobs(keywords=["data science", "data analyst", "data scientist", "software engineer"], pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        print(f"Searching for: {keyword}")
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            print(f"Scraping page {page} for keyword '{keyword}'")
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")

            job_blocks = soup.find_all("div", class_="ads-details")
            for job in job_blocks:
                try:
                    title = job.find("h4").get_text(strip=True)
                    company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                    location = job.find("p").get_text(strip=True)
                    experience = job.find("p", class_="emp-exp").get_text(strip=True)
                    key_skills_tag = job.find("span", string="Key Skills")
                    skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                    summary_tag = job.find("span", string="Summary")
                    summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                    jobs_list.append({
                        "Keyword": keyword,
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience,
                        "Summary": summary,
                        "Skills": skills
                    })
                except Exception as e:
                    print(f"Error parsing job block: {e}")
                    continue

            time.sleep(1)

    return pd.DataFrame(jobs_list)

def determine_optimal_clusters(skill_matrix, max_k=10):
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(skill_matrix)
        score = silhouette_score(skill_matrix, labels)
        silhouette_scores.append(score)
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return best_k

def assign_clusters_to_jobs(job_dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    skill_matrix = tfidf.fit_transform(job_dataframe['Skills'])
    optimal_k = determine_optimal_clusters(skill_matrix, max_k=10)
    clustering_model = KMeans(n_clusters=optimal_k, random_state=42)
    labels = clustering_model.fit_predict(skill_matrix)
    job_dataframe['Cluster'] = labels
    return job_dataframe, clustering_model, tfidf

def persist_models(model, tfidf_vectorizer, model_path='kmeans_model.pkl', vectorizer_path='vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

def load_models(model_path='kmeans_model.pkl', vectorizer_path='vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_job_cluster(skills_text, model, tfidf_vec):
    features = tfidf_vec.transform([skills_text])
    predicted_label = model.predict(features)
    return predicted_label[0]
