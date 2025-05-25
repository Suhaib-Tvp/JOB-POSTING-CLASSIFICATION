# app.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os

def scrape_karkidi_jobs(keywords=["data science", "data analyst"], pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
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
                    jobs_list.append({
                        "Keyword": keyword,
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience,
                        "Skills": skills
                    })
                except Exception:
                    continue
            time.sleep(1)
    return pd.DataFrame(jobs_list)

def train_and_save_model(df, n_clusters=3, model_path='kmeans_model.pkl', vectorizer_path='vectorizer.pkl'):
    tfidf = TfidfVectorizer(stop_words='english')
    skill_matrix = tfidf.fit_transform(df['Skills'].fillna(''))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(skill_matrix)
    joblib.dump(kmeans, model_path)
    joblib.dump(tfidf, vectorizer_path)
    return kmeans, tfidf

def load_models(model_path='kmeans_model.pkl', vectorizer_path='vectorizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_job_cluster(skills_text, model, tfidf_vec):
    features = tfidf_vec.transform([skills_text])
    predicted_label = model.predict(features)
    return predicted_label[0]
