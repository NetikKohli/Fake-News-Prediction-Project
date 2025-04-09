import requests
import re
import time
from datetime import datetime, timedelta

API_KEY = "3c9777379ed746e8be4d6bb228c85822"
PAGE_SIZE = 100

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

def fetch_real_news_for_date(date_str):
    all_texts = []
    for page in range(1, 6):
        articles = fetch_page(
            api_key=API_KEY,
            topic="technology OR politics OR science",
            from_date=date_str + "T00:00:00",
            to_date=date_str + "T23:59:59",
            page=page,
            domains="bbc.com,cnn.com,nytimes.com,reuters.com"
        )
        if not articles:
            break
        for art in articles:
            title = art.get("title") or ""
            desc  = art.get("description") or ""
            content = clean_text(title + " " + desc)
            if content:
                all_texts.append(content)
        if len(articles) < PAGE_SIZE:
            break
        time.sleep(1)
    return all_texts

def fetch_fake_news_for_date(date_str):
    all_texts = []
    for page in range(1, 6):
        articles = fetch_page(
            api_key=API_KEY,
            topic="hoax OR fake OR misinformation OR conspiracy OR rumor",
            from_date=date_str + "T00:00:00",
            to_date=date_str + "T23:59:59",
            page=page,
            domains=None
        )
        if not articles:
            break
        for art in articles:
            title = art.get("title") or ""
            desc  = art.get("description") or ""
            content = clean_text(title + " " + desc)
            if content:
                all_texts.append(content)
        if len(articles) < PAGE_SIZE:
            break
        time.sleep(1)
    return all_texts

if __name__ == "__main__":
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    real_texts = fetch_real_news_for_date(date_str)
    print(f"Fetched {len(real_texts)} real news articles for {date_str}")

    fake_texts = fetch_fake_news_for_date(date_str)
    print(f"Fetched {len(fake_texts)} fake news articles for {date_str}")
