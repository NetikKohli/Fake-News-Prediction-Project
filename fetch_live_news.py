#!/usr/bin/env python3
"""
fetch_live_news.py

This script fetches real and fake news headlines from NewsAPI, cleans the text,
stores them in CSV files, and runs predictions on all collected items to
produce a consolidated live_predictions.csv. Comments and structure have
been improved for clarity and maintainability.
"""

import os
import time
import re
import pickle
import pandas as pd
import requests

from datetime import datetime, timedelta

# Constants
API_KEY       = "3c9777379ed746e8be4d6bb228c85822"
PAGE_SIZE     = 100
REAL_CSV      = "real_news.csv"
FAKE_CSV      = "fake_news.csv"
PRED_CSV      = "live_predictions.csv"

# Load model and vectorizer once, to avoid reloading in loops
try:
    model       = pickle.load(open("best_model.pkl", "rb"))
    vectorizer  = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    model       = None
    vectorizer  = None
    print(f"[ERROR] Could not load model/vectorizer: {e}")

def clean_text(text):
    """
    Remove URLs, HTML tags, non-letter characters, and extra whitespace
    from a given text string. Return cleaned, lowercase text.
    """
    if not text:
        return ""
    # Lowercase and remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    # Strip HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z ]', '', text)
    # Collapse multiple spaces
    return re.sub(r'\\s+', ' ', text).strip()

def fetch_page(api_key, topic, from_date, to_date, page, domains=None):
    """
    Fetch a single page of articles from NewsAPI.

    Args:
        api_key (str): Your NewsAPI key.
        topic (str): Query keywords (e.g., "technology OR science").
        from_date (str): Start datetime in ISO format (YYYY-MM-DDThh:mm:ss).
        to_date (str): End datetime in ISO format (YYYY-MM-DDThh:mm:ss).
        page (int): Page number to retrieve.
        domains (str or None): Comma-separated list of domains to filter.

    Returns:
        list: A list of article dicts. Empty list on error or no more results.
    """
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
        # Log the error and return empty to stop paging
        print(f"[WARN] NewsAPI error on page {page}: {data.get('message', 'Unknown error')}")
        return []
    return data.get("articles", [])

def fetch_day(api_key, topic, date_str, domains=None):
    """
    Fetch all articles for a given date, cleaning text and collecting results.

    Args:
        api_key (str): Your NewsAPI key.
        topic (str): Query keywords.
        date_str (str): Date in YYYY-MM-DD format.
        domains (str or None): Optional domain filter.

    Returns:
        list of str: A list of cleaned article texts.
    """
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
            # Only keep if there’s at least 6 words after cleaning
            if len(content.split()) > 5:
                all_texts.append(content)

        # If fewer than PAGE_SIZE articles, we’ve reached the end
        if len(articles) < PAGE_SIZE:
            break

        # Rate‐limit courtesy pause
        time.sleep(1)

    return all_texts

def save_deduplicated_csv(filename, new_texts, label):
    """
    Append new_texts to filename as rows with the given label, removing duplicates.

    Args:
        filename (str): Path to the CSV file.
        new_texts (list of str): Cleaned text strings to save.
        label (int): 0 for real, 1 for fake.
    """
    new_df = pd.DataFrame({'text': new_texts, 'label': label})

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Drop exact-duplicate texts
    combined_df.drop_duplicates(subset="text", inplace=True)
    combined_df.to_csv(filename, index=False)
    print(f"[INFO] Saved {len(new_texts)} new articles to {filename} (Total rows: {len(combined_df)})")

def save_news_and_predict():
    """
    Core function to fetch real and fake articles for “today,” save them,
    then read all saved articles, predict with the model, and output live_predictions.csv.
    """
    if model is None or vectorizer is None:
        print("[ERROR] Cannot predict because model/vectorizer failed to load.")
        return

    today     = datetime.utcnow().date()
    date_str  = today.strftime("%Y-%m-%d")

    # 1) Fetch and save REAL news
    print(f"[INFO] Fetching REAL news for {date_str} …")
    real_texts = fetch_day(
        API_KEY,
        topic="technology OR politics OR science",
        date_str=date_str,
        domains="bbc.com,cnn.com,nytimes.com,reuters.com"
    )
    save_deduplicated_csv(REAL_CSV, real_texts, label=0)

    # 2) Fetch and save FAKE news
    print(f"[INFO] Fetching FAKE news for {date_str} …")
    fake_texts = fetch_day(
        API_KEY,
        topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
        date_str=date_str,
        domains=None
    )
    save_deduplicated_csv(FAKE_CSV, fake_texts, label=1)

    # 3) Combine both CSVs and run prediction
    combined_frames = []
    if os.path.exists(REAL_CSV):
        combined_frames.append(pd.read_csv(REAL_CSV))
    if os.path.exists(FAKE_CSV):
        combined_frames.append(pd.read_csv(FAKE_CSV))

    if not combined_frames:
        print("[WARN] No data to predict.")
        return

    df_all = pd.concat(combined_frames, ignore_index=True).drop_duplicates(subset="text")
    texts = df_all["text"].tolist()

    # Transform all texts and predict once
    vecs  = vectorizer.transform(texts)
    preds = model.predict(vecs)

    # Build output DataFrame
    df_all["prediction"]     = preds
    df_all["prediction_str"] = df_all["prediction"].apply(lambda x: "Real" if x == 0 else "Fake")

    # Save to live_predictions.csv
    df_all.to_csv(PRED_CSV, index=False)
    print(f"[INFO] Saved live predictions to {PRED_CSV} (Total rows: {len(df_all)})")

if __name__ == "__main__":
    save_news_and_predict()
