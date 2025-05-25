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

try:
    model      = pickle.load(open("best_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    model = None
    vectorizer = None
    print(f"[ERROR] Model/vectorizer load failed: {e}")

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

def fetch_page(api_key, topic, from_date, to_date, page, domains=None):
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": topic,
            "language": "en",
            "from": from_date,
            "to": to_date,
            "pageSize": PAGE_SIZE,
            "page": page,
            "sortBy": "publishedAt",
            "apiKey": api_key,
            **({"domains": domains} if domains else {})
        }
    )
    data = resp.json()
    return data.get("articles", []) if data.get("status") == "ok" else []

def fetch_day(api_key, topic, date_str, domains=None):
    texts = []
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
                texts.append(content)
        if len(articles) < PAGE_SIZE:
            break
        time.sleep(1)
    return texts

def fetch_range(api_key, topic, start_date, end_date, domains=None):
    curr_date = start_date
    collected = []
    while curr_date <= end_date:
        date_str = curr_date.strftime("%Y-%m-%d")
        day_texts = fetch_day(api_key, topic, date_str, domains=domains)
        collected.extend(day_texts)
        curr_date += timedelta(days=1)
        time.sleep(1)
    return list(dict.fromkeys(collected))  # Deduplicate by exact match

def save_deduplicated_csv(filename, new_texts, label):
    new_df = pd.DataFrame({'text': new_texts, 'label': label})
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset='text', inplace=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filename, index=False)

def save_news_and_predict():
    if model is None or vectorizer is None:
        print("[ERROR] Model/vectorizer unavailable.")
        return

    today      = datetime.utcnow().date()
    start_date = today - timedelta(days=29)

    real_texts = fetch_range(
        API_KEY,
        topic="technology OR politics OR science",
        start_date=start_date,
        end_date=today,
        domains="bbc.com,cnn.com,nytimes.com,reuters.com"
    )
    save_deduplicated_csv(REAL_CSV, real_texts, label=0)

    fake_texts = fetch_range(
        API_KEY,
        topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
        start_date=start_date,
        end_date=today
    )
    save_deduplicated_csv(FAKE_CSV, fake_texts, label=1)

    df_list = []
    for f in [REAL_CSV, FAKE_CSV]:
        if os.path.exists(f):
            df_list.append(pd.read_csv(f))
    if not df_list:
        print("[WARN] No data to predict.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates(subset="text")
    texts  = df_all["text"].tolist()
    vecs   = vectorizer.transform(texts)
    preds  = model.predict(vecs)
    df_all["prediction"]     = preds
    df_all["prediction_str"] = df_all["prediction"].apply(lambda x: "Real" if x == 0 else "Fake")
    df_all.to_csv(PRED_CSV, index=False)

if __name__ == "__main__":
    save_news_and_predict()
