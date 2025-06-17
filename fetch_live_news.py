import requests
import pandas as pd
import re
import time
from datetime import datetime, timedelta
import os 

# Constants
PAGE_SIZE = 100
REAL_CSV = "real_news.csv"
FAKE_CSV = "fake_news.csv"

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def fetch_page(api_key, topic, from_date, to_date, page, domains=None):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "from": from_date,
        "to": to_date,
        "pageSize": PAGE_SIZE,
        "page": page,
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    if domains:
        params["domains"] = domains

    resp = requests.get(url, params=params)
    data = resp.json()

    if data.get("status") != "ok":
        print(f"Error on page {page} for {from_date} to {to_date}: {data.get('message', 'Unknown error')}")
        return []

    return data.get("articles", [])

def fetch_day(api_key, topic, date_str, domains=None):
    all_texts = []
    for page in range(1, 6):
        articles = fetch_page(
            api_key=api_key,
            topic=topic,
            from_date=date_str + "T00:00:00",
            to_date=date_str + "T23:59:59",
            page=page,
            domains=domains
        )
        if not articles:
            break
        for art in articles:
            title = art.get("title") or ""
            desc  = art.get("description") or ""
            content = clean_text(title + " " + desc)
            if len(content.split()) > 5:
                all_texts.append(content)
        if len(articles) < PAGE_SIZE:
            break
        time.sleep(1)
    return all_texts

def fetch_range(api_key, topic, start_date, end_date, domains=None):
    curr_date = start_date
    collected = []

    while curr_date <= end_date:
        date_str = curr_date.strftime("%Y-%m-%d")
        print(f"Fetching for {date_str} ...", end=" ")
        day_texts = fetch_day(api_key, topic, date_str, domains=domains)
        print(f"{len(day_texts)} articles")
        collected.extend(day_texts)
        curr_date += timedelta(days=1)
        time.sleep(1)

    unique_texts = list(dict.fromkeys(collected))  # Deduplicate by exact match
    return unique_texts

def save_deduplicated_csv(filename, new_texts, label):
    new_df = pd.DataFrame({'text': new_texts, 'label': label})
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset='text', inplace=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filename, index=False)
    print(f"Saved {len(new_texts)} new articles to {filename} (Total: {len(combined_df)})")

def save_news_to_csv():
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=29)

    print("=== Fetching REAL news from multiple days ===")
    real_texts = fetch_range(
        API_KEY,
        topic="technology OR politics OR science",
        start_date=start_date,
        end_date=end_date,
        domains="bbc.com,cnn.com,nytimes.com,reuters.com"
    )
    save_deduplicated_csv(REAL_CSV, real_texts, label=0)

    print("\n Fetching FAKE news (hoaxes) from multiple days ")
    fake_texts = fetch_range(
        API_KEY,
        topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
        start_date=start_date,
        end_date=end_date,
        domains=None
    )
    save_deduplicated_csv(FAKE_CSV, fake_texts, label=1)

if __name__ == "__main__":
    save_news_to_csv()
