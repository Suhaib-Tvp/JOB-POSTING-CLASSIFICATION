import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_karkidi_jobs(keywords=["data science"], pages=1):
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
                    location = job.find("small", class_="text-muted").get_text(strip=True)
                    date_posted = job.find("span", class_="date").get_text(strip=True)
                    link = "https://www.karkidi.com" + job.find("a", href=True)["href"]
                    jobs_list.append({
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Date Posted": date_posted,
                        "Link": link
                    })
                except Exception:
                    continue
            time.sleep(1)  # Be nice to the server

    return pd.DataFrame(jobs_list)

# Streamlit App
st.title("Karkidi Job Scraper")

keywords_input = st.text_input("Enter job keywords (comma-separated):", "data science, data analyst")
pages = st.slider("Number of pages to scrape per keyword:", 1, 5, 1)

if st.button("Scrape Jobs"):
    with st.spinner("Scraping in progress..."):
        keywords = [kw.strip() for kw in keywords_input.split(",")]
        results_df = scrape_karkidi_jobs(keywords, pages)
        st.success(f"Scraped {len(results_df)} job listings.")
        st.dataframe(results_df)

        # Download link
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='karkidi_jobs.csv',
            mime='text/csv',
        )
