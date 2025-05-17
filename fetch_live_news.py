#!/usr/bin/env python3
import os
import time
import re
import pickle
import pandas as pd
import requests
from datetime import datetime, timedelta

API_KEY    = "3c9777379ed746e8be4d6bb228c85822"
PAGE_SIZE  = 100
REAL_CSV   = "real_news.csv"
FAKE_CSV   = "fake_news.csv"
PRED_CSV   = "live_predictions.csv"

# Load model and vectorizer
try:
    model      = pickle.load(open("best_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    model      = None
    vectorizer = None
    print(f"[ERROR] Could not load model/vectorizer: {e}")

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
        return []
    return data.get("articles", [])

def fetch_day(api_key, topic, date_str, domains=None):
    all_texts = []
    for page in range(1, 6):
        articles = fetch_page(
            api_key=api_key,
            topic=topic,
            from_date=f"{date_str}T00:00:00",
            to_date=f"{date_str}T23:59:59",
            page=page,
            domains=domains
        )
        if not articles:
            break

        for art in articles:
            title   = art.get("title") or ""
            desc    = art.get("description") or ""
            content = clean_text(f"{title} {desc}")
            if len(content.split()) > 5:
                all_texts.append(content)

        if len(articles) < PAGE_SIZE:
            break

        time.sleep(1)

    return all_texts

def save_deduplicated_csv(filename, new_texts, label):
    new_df = pd.DataFrame({'text': new_texts, 'label': label})
    if os.path.exists(filename):
        existing_df  = pd.read_csv(filename)
        combined_df  = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.drop_duplicates(subset="text", inplace=True)
    combined_df.to_csv(filename, index=False)
    print(f"[INFO] Saved {len(new_texts)} new articles to {filename} (Total rows: {len(combined_df)})")

def save_news_and_predict():
    if model is None or vectorizer is None:
        print("[ERROR] Cannot predict because model/vectorizer failed to load.")
        return

    today    = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    real_texts = fetch_day(
        API_KEY,
        topic="technology OR politics OR science",
        date_str=date_str,
        domains="bbc.com,cnn.com,nytimes.com,reuters.com"
    )
    save_deduplicated_csv(REAL_CSV, real_texts, label=0)

    fake_texts = fetch_day(
        API_KEY,
        topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
        date_str=date_str,
        domains=None
    )
    save_deduplicated_csv(FAKE_CSV, fake_texts, label=1)

    combined_frames = []
    if os.path.exists(REAL_CSV):
        combined_frames.append(pd.read_csv(REAL_CSV))
    if os.path.exists(FAKE_CSV):
        combined_frames.append(pd.read_csv(FAKE_CSV))

    if not combined_frames:
        print("[WARN] No data to predict.")
        return

    df_all = pd.concat(combined_frames, ignore_index=True).drop_duplicates(subset='text')
    texts  = df_all["text"].tolist()
    vecs   = vectorizer.transform(texts)
    preds  = model.predict(vecs)
    df_all["prediction"]     = preds
    df_all["prediction_str"] = df_all["prediction"].apply(lambda x: "Real" if x == 0 else "Fake")
    df_all.to_csv(PRED_CSV, index=False)
    print(f"[INFO] Saved live predictions to {PRED_CSV} (Total rows: {len(df_all)})")

if __name__ == "__main__":
    save_news_and_predict()
