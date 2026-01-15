# ==============================
# 📦 IMPORTS & CONFIGURATIE
# ==============================

import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
# ==============================
# ▶️ CEL 4 AANROEPEN
# ==============================

mijn_tweets = fetch_my_tweets(USERNAME, max_tweets=30)
print(f"🔎 Debug: mijn_tweets bevat {len(mijn_tweets)} tweets")

# ==============================
# 🔄 CEL 5 — DATA COMBINEREN
# ==============================

print("\n🔄 DATA COMBINEREN")
print("=" * 60)

# Combineer oude + nieuwe data
if oude_data.empty:
    combined = mijn_tweets.copy()
    print(f"📊 Alleen nieuwe tweets: {len(combined)}")
else:
    combined = pd.concat([oude_data, mijn_tweets], ignore_index=True)
    print(f"📊 Oude: {len(oude_data)} + Nieuw: {len(mijn_tweets)}")

# Duplicaten verwijderen op tweet-id
if "id" in combined.columns:
    before = len(combined)
    combined = combined.drop_duplicates(subset="id", keep="first")
    removed = before - len(combined)
    if removed > 0:
        print(f"🔍 {removed} duplicaten verwijderd")

# Tijdzone fix
if "tijd" in combined.columns:
    combined["tijd"] = pd.to_datetime(combined["tijd"]).dt.tz_localize(None)

# Sorteren
combined = combined.sort_values("tijd", ascending=False).reset_index(drop=True)

print(f"\n✅ Totaal tweets: {len(combined)}")
if not combined.empty:
    print(f"📅 Van {combined['tijd'].min()} tot {combined['tijd'].max()}")

# ==============================
# 🛠️ CEL 6 — HELPER FUNCTIES
# ==============================

def extract_hashtags(text):
    """Haalt hashtags uit tekst"""
    return re.findall(r"#(\w+)", str(text).lower())


def categorize_content(text):
    """Categoriseert content type op basis van tekst"""
    text_lower = str(text).lower()

    if any(word in text_lower for word in ["vraag", "?", "poll", "question"]):
        return "vraag/poll"
    elif any(word in text_lower for word in ["tip", "advies", "hoe", "guide", "how"]):
        return "educatief"
    elif any(word in text_lower for word in ["nieuw", "new", "launch", "dropping"]):
        return "aankondiging"
    elif any(word in text_lower for word in ["dank", "thanks", "appreciate"]):
        return "interactie"
    elif any(word in text_lower for word in ["link", "bio", "check", "subscribe"]):
        return "promotie"
    else:
        return "algemeen"


print("✅ Helper functies geladen")

# ==============================
# 🧩 CEL 7 (MINI) — FEATURES VOOR AI
# ==============================

if combined.empty:
    print("⚠️ Geen data voor feature engineering")
else:
    combined["uur"] = combined["tijd"].dt.hour
    combined["dag"] = combined["tijd"].dt.dayofweek
    combined["aantal_hashtags"] = combined["text"].apply(
        lambda x: len(extract_hashtags(x))
    )
    combined["tekst_lengte"] = combined["text"].astype(str).str.len()

    # Boolean features naar int (handig voor ML)
    if "heeft_media" in combined.columns:
        combined["heeft_media"] = combined["heeft_media"].astype(int)
    else:
        combined["heeft_media"] = 0

    if "heeft_link" in combined.columns:
        combined["heeft_link"] = combined["heeft_link"].astype(int)
    else:
        combined["heeft_link"] = 0

    print("✅ Feature engineering voor AI voltooid")

# ==============================
# 🧠 CONTENT & FORMAT ADVIES (HYBRIDE)
# ==============================

print("\n" + "=" * 60)
print("📝 CONTENT & FORMAT ADVIES")
print("=" * 60)

n_tweets = len(combined)

# ------------------------------
# MEDIA ADVIES
# ------------------------------
if "heeft_media" in combined.columns:
    media_pct = combined["heeft_media"].mean()

    print("\n🖼️ MEDIA")
    print("-" * 60)
    print(f"📊 {media_pct:.0%} van je tweets bevat media")

    if n_tweets < 5:
        print("⚠️ Weinig data — advies is indicatief")
    else:
        tweets_met_media = combined[combined["heeft_media"] == 1]
        tweets_zonder_media = combined[combined["heeft_media"] == 0]

        if not tweets_met_media.empty and not tweets_zonder_media.empty:
            eng_media = tweets_met_media["total_engagement"].mean()
            eng_no_media = tweets_zonder_media["total_engagement"].mean()

            if eng_media > eng_no_media:
                print("✅ Tweets met media presteren beter dan zonder media")
            else:
                print("✅ Tweets zonder media presteren beter dan met media")

    if media_pct < 0.5:
        print("💡 Overweeg vaker media (foto/video) te gebruiken")
    else:
        print("💡 Je mediagebruik zit goed")

# ------------------------------
# HASHTAG ADVIES
# ------------------------------
if "aantal_hashtags" in combined.columns:
    avg_hash = combined["aantal_hashtags"].mean()

    print("\n🏷️ HASHTAGS")
    print("-" * 60)
    print(f"📊 Gemiddeld {avg_hash:.1f} hashtags per tweet")

    if avg_hash < 1:
        print("💡 Je gebruikt weinig hashtags — probeer er 1–2 toe te voegen")
    elif avg_hash > 5:
        print("💡 Je gebruikt veel hashtags — minder kan soms beter werken")
    else:
        print("✅ Je hashtag‑gebruik zit in een gezonde range")

    # Top hashtags (inhoudelijk)
    if "text" in combined.columns:
        from collections import Counter
        all_tags = []
        for t in combined["text"]:
            all_tags.extend(extract_hashtags(t))

        if all_tags:
            top_tags = Counter(all_tags).most_common(5)
            print("\n🔥 Meest gebruikte hashtags:")
            for tag, cnt in top_tags:
                print(f"   #{tag} ({cnt}×)")

# ==============================
# 🤖 CEL 8 — AI VOORSPELLINGEN
# ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_personal_ai_model(df):
    print("=" * 60)
    print("🤖 AI MODEL TRAINEN")
    print("=" * 60)

    # Check engagement data
    use_engagement = False
    if "likes" in df.columns:
        tweets_met_eng = df[
            (df["likes"] > 0) |
            (df["retweets"] > 0) |
            (df["replies"] > 0)
        ]

        if len(tweets_met_eng) >= 10:
            df = tweets_met_eng
            use_engagement = True

    # ==============================
    # FEATURE SET
    # ==============================
    X = df[[
        "uur",
        "dag",
        "aantal_hashtags",
        "tekst_lengte",
        "heeft_media",
        "heeft_link"
    ]].copy()

    # ==============================
    # TARGET
    # ==============================
    if use_engagement:
        y = df["total_engagement"]
        print("📊 Methode: engagement-based AI")
    else:
        print("📊 Methode: frequency-based AI")

        freq = df.groupby(["uur", "dag"]).size()
        max_freq = freq.max()

        def freq_score(row):
            return freq.get((row["uur"], row["dag"]), 1) / max_freq

        y = df.apply(freq_score, axis=1)

    # ==============================
    # MODEL TRAINEN
    # ==============================
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    print("✅ Model getraind")

    # ==============================
    # VOORSPELLINGEN
    # ==============================
    predictions = []

    avg = {
        "aantal_hashtags": df["aantal_hashtags"].mean(),
        "tekst_lengte": df["tekst_lengte"].mean(),
        "heeft_media": df["heeft_media"].mean(),
        "heeft_link": df["heeft_link"].mean()
    }

    for dag in range(7):
        for uur in range(6, 24):
            row = [
                uur,
                dag,
                avg["aantal_hashtags"],
                avg["tekst_lengte"],
                avg["heeft_media"],
                avg["heeft_link"]
            ]
            score = model.predict([row])[0]
            predictions.append({
                "dag": dag,
                "uur": uur,
                "score": score
            })

    pred_df = pd.DataFrame(predictions)

    # ==============================
    # TOP MOMENTEN
    # ==============================
    top = pred_df.nlargest(10, "score")

    dagen_map = {
        0: "Maandag",
        1: "Dinsdag",
        2: "Woensdag",
        3: "Donderdag",
        4: "Vrijdag",
        5: "Zaterdag",
        6: "Zondag"
    }

    print("\n🏆 TOP 10 AANBEVOLEN POSTMOMENTEN")
    print("-" * 60)
    for i, r in enumerate(top.itertuples(), 1):
        print(
            f"{i:2d}. {dagen_map[r.dag]:9s} om {r.uur:02d}:00 "
            f"(score: {r.score*100:.0f}%)"
        )

    return model, pred_df

# ==============================
# ▶️ CEL 8 AANROEPEN
# ==============================

if combined.empty or len(combined) < 3:
    print("⚠️ Te weinig data voor AI")
    model = None
    predictions = pd.DataFrame()
else:
    model, predictions = train_personal_ai_model(combined)

# ==============================
# 💾 CEL 10 — DATA OPSLAAN
# ==============================

if combined.empty:
    print("⚠️ Geen data om op te slaan")
else:
    run_date = datetime.now().strftime("%Y-%m-%d")
    safe_username = USERNAME.lower().replace("@", "")
    filename = f"{safe_username}_{run_date}.xlsx"

    path = os.path.join("data", filename)
    combined.to_excel(path, index=False)

    print(f"✅ Data opgeslagen: {path}")
    print(f"📊 {len(combined)} tweets opgeslagen")

# ==============================
# 💾 DATA COMMITTEN NAAR GITHUB
# ==============================

import subprocess

try:
    subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "actions@github.com"], check=True)

    subprocess.run(["git", "add", "data/*.xlsx"], check=True)
    subprocess.run(["git", "commit", "-m", f"Update data {run_date}"], check=True)
    subprocess.run(["git", "push"], check=True)

    print("✅ Data succesvol opgeslagen in GitHub repository")

except Exception as e:
    print(f"⚠️ Kon data niet committen: {e}")
