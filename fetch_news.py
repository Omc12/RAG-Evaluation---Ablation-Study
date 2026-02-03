import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def fetch_news(ticker, limit=30):
    if not API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables. Please set it in your .env file.")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": API_KEY
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if "feed" not in data:
        raise RuntimeError(f"API error or rate limit: {data}")

    articles = []
    for item in data["feed"]:
        text = item.get("summary", "").strip()

        if len(text) < 80:
            continue

        articles.append({
            "title": item.get("title"),
            "text": text,
            "url": item.get("url"),
            "time": item.get("time_published"),
            "sentiment": item.get("overall_sentiment_label")
        })

    return articles
