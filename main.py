# Not needed for Streamlit, but can help testing logic separately
from app import scrape_karkidi_jobs

if __name__ == "__main__":
    keywords = ["data science", "data analyst"]
    df = scrape_karkidi_jobs(keywords, pages=1)
    print(df.head())
