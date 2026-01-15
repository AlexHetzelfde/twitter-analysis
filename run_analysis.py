# ==============================
# 📦 IMPORTS & CONFIGURATIE
# ==============================

import pandas as pd
import os
from datetime import datetime

# ⚙️ CONFIGURATIE
USERNAME = "Oliviaafairy"

print("=" * 60)
print("🧚 TWITTER AI ANALYSE")
print("=" * 60)
print(f"✅ Script gestart voor @{USERNAME}")
print(f"📅 Run datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ==============================
# 🔑 TWITTER API SETUP
# ==============================

import tweepy

print("\n🔑 Twitter API verbinden...")

client = tweepy.Client(
    bearer_token=os.environ["X_BEARER_TOKEN"],
    consumer_key=os.environ["X_API_KEY"],
    consumer_secret=os.environ["X_API_SECRET"],
    access_token=os.environ["X_ACCESS_TOKEN"],
    access_token_secret=os.environ["X_ACCESS_SECRET"],
    wait_on_rate_limit=True
)

me = client.get_me()
print(f"✅ Verbonden als @{me.data.username}")

# ==============================
# 📂 OUDE DATA AUTOMATISCH LADEN
# ==============================

DATA_DIR = "data"

def load_previous_data():
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".xlsx")
    ]

    if not files:
        print("ℹ️ Geen oude data gevonden (eerste run)")
        return pd.DataFrame()

    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f))
    )

    path = os.path.join(DATA_DIR, latest_file)
    print(f"📂 Oude data geladen: {path}")

    df = pd.read_excel(path)

    if "tijd" in df.columns:
        df["tijd"] = pd.to_datetime(df["tijd"])

    print(f"✅ {len(df)} rijen uit vorige run")
    return df

oude_data = load_previous_data()

# ==============================
# 🆕 JOUW TWEETS OPHALEN
# ==============================

import re

TWEETS_OPGEHAALD_DEZE_MAAND = 0

def fetch_my_tweets(username, max_tweets=10):
    global TWEETS_OPGEHAALD_DEZE_MAAND

    print("\n📊 JOUW TWEETS OPHALEN")
    print("=" * 60)

    query = f"from:{username} -is:retweet -is:reply"

    response = client.search_recent_tweets(
        query=query,
        max_results=max_tweets,
        tweet_fields=["created_at", "text", "public_metrics", "attachments"],
        expansions=["attachments.media_keys"],
        media_fields=["type"]
    )

    if not response.data:
        print("⚠️ Geen tweets gevonden")
        return pd.DataFrame()

    media_dict = {}
    if hasattr(response, "includes") and "media" in response.includes:
        for media in response.includes["media"]:
            media_dict[media.media_key] = media.type

    rows = []
    for t in response.data:
        metrics = t.public_metrics or {}

        rows.append({
            "id": str(t.id),
            "tijd": pd.to_datetime(t.created_at),
            "text": t.text,
            "likes": metrics.get("like_count", 0),
            "retweets": metrics.get("retweet_count", 0),
            "replies": metrics.get("reply_count", 0),
            "quotes": metrics.get("quote_count", 0),
        })

    df = pd.DataFrame(rows)
    df["total_engagement"] = (
        df["likes"] + df["retweets"] + df["replies"] + df["quotes"]
    )

    print(f"✅ {len(df)} tweets opgehaald")
    print(f"📅 Van {df['tijd'].min()} tot {df['tijd'].max()}")

    return df
    mijn_tweets = fetch_my_tweets(USERNAME, max_tweets=30)
