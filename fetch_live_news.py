import requests
import pandas as pd
import re
import time
import pickle
from datetime import datetime, timedelta
import os

API_KEY = "3c9777379ed746e8be4d6bb228c85822"
PAGE_SIZE = 100
REAL_CSV = "real_news.csv"
FAKE_CSV = "fake_news.csv"
PRED_CSV = "live_predictions.csv"

# Load model and vectorizer from Phase 1
model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

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
        print(f"Error on page {page}: {data.get('message', 'Unknown error')}")
        return []
    return data.get("articles", [])

def fetch_day(api_key, topic, date_str, domains=None):
    all_articles = []
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
                all_articles.append({
                    "text": content,
                    "publishedAt": art.get("publishedAt", "")
                })
        if len(articles) < PAGE_SIZE:
            break
        time.sleep(1)
    return all_articles

def save_deduplicated_csv(filename, df_new):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        combined_df = df_new
    combined_df.drop_duplicates(subset='text', inplace=True)
    combined_df.to_csv(filename, index=False)
    print(f"Saved to {filename} (Total: {len(combined_df)})")

def save_news_and_predict():
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    # Fetch real news
    real_articles = fetch_day(
        API_KEY,
        topic="technology OR politics OR science",
        date_str=date_str,
        domains="bbc.com,cnn.com,nytimes.com,reuters.com"
    )
    if real_articles:
        df_real = pd.DataFrame(real_articles)
        df_real["label"] = 0
        save_deduplicated_csv(REAL_CSV, df_real[["text", "label"]])

    # Fetch fake news
    fake_articles = fetch_day(
        API_KEY,
        topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
        date_str=date_str,
        domains=None
    )
    if fake_articles:
        df_fake = pd.DataFrame(fake_articles)
        df_fake["label"] = 1
        save_deduplicated_csv(FAKE_CSV, df_fake[["text", "label"]])

    # Combine and predict on all new texts
    combined = []
    if os.path.exists(REAL_CSV):
        combined.append(pd.read_csv(REAL_CSV))
    if os.path.exists(FAKE_CSV):
        combined.append(pd.read_csv(FAKE_CSV))
    if combined:
        df_all = pd.concat(combined, ignore_index=True).drop_duplicates(subset='text')
        texts = df_all["text"].tolist()
        vecs = vectorizer.transform(texts)
        preds = model.predict(vecs)
        df_all["prediction"] = preds
        df_all["prediction_str"] = df_all["prediction"].apply(lambda x: "Real" if x == 0 else "Fake")
        df_all.to_csv(PRED_CSV, index=False)
        print(f"Saved live predictions to {PRED_CSV} (Total: {len(df_all)})")

if __name__ == "__main__":
    save_news_and_predict()
