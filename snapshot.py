import json
from fetch_news import fetch_news

news = fetch_news("TSLA", limit=25)

if len(news) < 5:
    raise RuntimeError("Insufficient data for evaluation")

with open("dataset_snapshot.json", "w") as f:
    json.dump(news, f, indent=2)

print(f"Snapshot saved with {len(news)} articles")
