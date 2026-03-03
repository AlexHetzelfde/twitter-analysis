# ==============================
# 📦 IMPORTS & CONFIGURATIE
# ==============================

import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ⚙️ CONFIGURATIE
USERNAME = "YourSlutHaley"

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

    # ✅ DIT MOET IN DE FUNCTIE
    for t in response.data:
        metrics = t.public_metrics or {}

        heeft_media = False
        media_type = None

        if hasattr(t, "attachments") and t.attachments:
            media_keys = t.attachments.get("media_keys", [])
            if media_keys:
                heeft_media = True
                media_type = media_dict.get(media_keys[0], "unknown")

        rows.append({
            "id": str(t.id),
            "tijd": pd.to_datetime(t.created_at),
            "text": t.text,
            "heeft_media": heeft_media,
            "media_type": media_type,
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

# ------------------------------
# DEDUPLICATIE (ROBUST)
# ------------------------------
if "id" in combined.columns:
    combined["id"] = combined["id"].astype(str)
    before = len(combined)
    combined = combined.drop_duplicates(subset="id", keep="first")
    removed = before - len(combined)
    if removed > 0:
        print(f"🔍 {removed} duplicaten verwijderd")

# Tijdzone fix
combined["tijd"] = pd.to_datetime(
    combined["tijd"],
    utc=True,
    errors="coerce"
).dt.tz_convert(None)

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
# 🧩 CEL 7 — ADVANCED FEATURE ENGINEERING
# ==============================

combined["uur"] = combined["tijd"].dt.hour
combined["dag"] = combined["tijd"].dt.dayofweek

# Basis
combined["tekst_lengte"] = combined["text"].astype(str).str.len()
combined["woordenaantal"] = combined["text"].astype(str).apply(lambda x: len(x.split()))
combined["aantal_hashtags"] = combined["text"].apply(lambda x: len(extract_hashtags(x)))

# Hashtag density
combined["hashtag_density"] = combined["aantal_hashtags"] / combined["woordenaantal"].replace(0, 1)

# Vraagvorm
combined["is_vraag"] = combined["text"].astype(str).str.contains(r"\?").astype(int)

# Uitroeptekens
combined["aantal_uitroep"] = combined["text"].astype(str).str.count("!")

# Hoofdletters percentage
def caps_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    caps = sum(1 for c in text if c.isupper())
    return caps / len(text)

combined["caps_ratio"] = combined["text"].apply(caps_ratio)

# Emoji count (ruwe detectie)
combined["emoji_count"] = combined["text"].astype(str).str.count(r"[^\w\s,]")

# Link detectie
combined["heeft_link"] = combined["text"].str.contains(r"https?://", case=False, na=False).astype(int)

# Content categorie
combined["content_type"] = combined["text"].apply(categorize_content)
combined = pd.get_dummies(combined, columns=["content_type"], drop_first=True)

# Media fix
combined["heeft_media"] = combined["heeft_media"].fillna(False).astype(int)

print("✅ Advanced features toegevoegd")

# ==============================
# 🧠 DIEPGAANDE PERFORMANCE ANALYSE
# ==============================

print("\n" + "=" * 60)
print("🧠 DIEPGAANDE CONTENT ANALYSE")
print("=" * 60)

if len(combined) >= 10:

    # Percentielen
    top_threshold = combined["total_engagement"].quantile(0.75)
    bottom_threshold = combined["total_engagement"].quantile(0.25)

    top = combined[combined["total_engagement"] >= top_threshold]
    bottom = combined[combined["total_engagement"] <= bottom_threshold]

    print("\n🔥 TOP 25% vs ❄️ BOTTOM 25%")
    print("-" * 60)

    def compare_feature(feature):
        if feature in combined.columns:
            print(f"{feature}:")
            print(f"   Top: {top[feature].mean():.2f}")
            print(f"   Bottom: {bottom[feature].mean():.2f}")

    for f in [
        "aantal_hashtags",
        "tekst_lengte",
        "woordenaantal",
        "heeft_media",
        "heeft_link",
        "is_vraag"
    ]:
        compare_feature(f)

    # ==============================
    # HASHTAG PERFORMANCE
    # ==============================
    print("\n🏷️ HASHTAG PERFORMANCE")
    print("-" * 60)

    baseline = combined["total_engagement"].mean()

    hashtag_stats = {}

    for _, row in combined.iterrows():
        tags = extract_hashtags(row["text"])
        for tag in tags:
            hashtag_stats.setdefault(tag, []).append(row["total_engagement"])

    results = []
    for tag, values in hashtag_stats.items():
        if len(values) >= 3:
            avg_eng = sum(values) / len(values)
            uplift = ((avg_eng - baseline) / baseline) * 100
            results.append((tag, len(values), avg_eng, uplift))

    results.sort(key=lambda x: x[3], reverse=True)

    for tag, count, avg_eng, uplift in results[:10]:
        print(f"#{tag} ({count}x) → {uplift:+.1f}% uplift")

else:
    print("⚠️ Te weinig data voor diepgaande analyse")

# ==============================
# 🧹 DATA OPSCHONEN VOOR AI
# ==============================

# Zorg dat hashtag-aantallen altijd geldig zijn
if "aantal_hashtags" in combined.columns:
    combined["aantal_hashtags"] = combined["aantal_hashtags"].fillna(0)

# Zorg dat IDs altijd strings zijn (voorkomt duplicaten)
if "id" in combined.columns:
    combined["id"] = combined["id"].astype(str)

# ==============================
# 🤖 CEL 8 — VERBETERDE AI VOORSPELLINGEN
# ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

def train_personal_ai_model(df):

    print("=" * 60)
    print("🤖 ADVANCED AI MODEL TRAINEN (MET VALIDATIE)")
    print("=" * 60)

    df = df[df["total_engagement"] > 0].copy()

    if len(df) < 8:
        print("⚠️ Te weinig data voor betrouwbaar model")
        return None, df

    y = np.log1p(df["total_engagement"])

    feature_cols = [
        "uur",
        "dag",
        "aantal_hashtags",
        "tekst_lengte",
        "woordenaantal",
        "hashtag_density",
        "is_vraag",
        "aantal_uitroep",
        "caps_ratio",
        "emoji_count",
        "heeft_media",
        "heeft_link"
    ]

    content_cols = [c for c in df.columns if c.startswith("content_type_")]
    feature_cols.extend(content_cols)

    X = df[feature_cols]

    # ==========================
    # Train / Test split
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==========================
    # Evaluatie
    # ==========================
    y_pred_test = model.predict(X_test)

    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test))

    print(f"📊 R² score: {r2:.3f}")
    print(f"📉 MAE (gem. fout in engagement): {mae:.2f}")

    # Cross validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"🔁 Cross-val gemiddelde R²: {cv_scores.mean():.3f}")

    # ==========================
    # Voorspellingen voor ALLE tweets
    # ==========================
    df["ai_prediction"] = np.expm1(model.predict(X))
    df["prediction_error"] = df["total_engagement"] - df["ai_prediction"]

    print("\n📈 TOP UNDERPERFORMERS (gemiste kansen)")
    under = df.sort_values("prediction_error").head(5)
    for _, row in under.iterrows():
        print(f"- {row['text'][:60]} | -{row['prediction_error']:.1f}")

    print("\n🚀 TOP OVERPERFORMERS")
    over = df.sort_values("prediction_error", ascending=False).head(5)
    for _, row in over.iterrows():
        print(f"- {row['text'][:60]} | +{row['prediction_error']:.1f}")

    # ==========================
    # Feature importance
    # ==========================
    importances = model.feature_importances_

        importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
    )
    # ==========================
    # Grotere simulatie ruimte
    # ==========================
    print("\n🧪 STRATEGISCHE SIMULATIE")
    print("-" * 60)

    best_score = -1
    best_combo = None

    avg_vals = X.mean()

    for uur in range(0, 24, 3):
        for hashtags in [0, 1, 2, 3]:
            for media in [0, 1]:
                test_row = avg_vals.copy()
                test_row["uur"] = uur
                test_row["aantal_hashtags"] = hashtags
                test_row["heeft_media"] = media

                pred = np.expm1(model.predict(test_row.to_frame().T)[0])

                if pred > best_score:
                    best_score = pred
                    best_combo = (uur, hashtags, media)

    print("Beste combinatie:")
    print(f"Tijdstip: {best_combo[0]}:00")
    print(f"Hashtags: {best_combo[1]}")
    print(f"Media: {'Ja' if best_combo[2] else 'Nee'}")
    print(f"Geschatte engagement: {best_score:.1f}")

    # ==========================================================
    # 📅 OPTIMAAL PUBLICATIEMOMENT
    # ==========================================================

    from datetime import timedelta

    print("\n📅 OPTIMAAL PUBLICATIEMOMENT")
    print("-" * 60)

    best_hour = best_combo[0]

    today = datetime.now()
    next_post_date = today.replace(hour=best_hour, minute=0, second=0, microsecond=0)

    if next_post_date < today:
        next_post_date += timedelta(days=1)

    print(f"Aanbevolen publicatie: {next_post_date.strftime('%Y-%m-%d %H:%M')}")

    # ==========================================================
    # 🧩 IDEALE POST STRUCTUUR
    # ==========================================================

    print("\n🧩 IDEALE POST STRUCTUUR")
    print("-" * 60)

    best_hashtags = best_combo[1]
    best_media = best_combo[2]

    avg_top_words = int(top["woordenaantal"].mean())
    avg_top_emoji = int(top["emoji_count"].mean())

    print(f"• Woordenaantal: ±{avg_top_words}")
    print(f"• Hashtags: {best_hashtags}")
    print(f"• Media: {'Ja' if best_media else 'Nee'}")
    print(f"• Emoji’s: {avg_top_emoji}")

    # ==========================================================
    # 🏷️ AANBEVOLEN HASHTAGS
    # ==========================================================

    print("\n🏷️ AANBEVOLEN HASHTAGS")
    print("-" * 60)

    if 'results' in globals() or 'results' in locals():
        best_tags = [r[0] for r in results[:best_hashtags]]

        for tag in best_tags:
            print(f"#{tag}")
    else:
        best_tags = []

    # ==========================================================
    # 📝 VOORBEELD POST (PATROON-GEBASEERD)
    # ==========================================================

    print("\n📝 VOORBEELD POST (AI-GEBASEERD)")
    print("-" * 60)

    top_texts = " ".join(top["text"].astype(str))

    common_words = (
        pd.Series(top_texts.lower().split())
        .value_counts()
        .head(15)
        .index.tolist()
    )

    hook_words = [w for w in common_words if len(w) > 4][:3]

    hashtag_string = " ".join(["#" + t for t in best_tags])

    generated_text = (
        f"{' '.join(hook_words).capitalize()}... {hashtag_string}"
    )

    print(generated_text)

    # ==========================================================
    # 🚀 AI CONTENT DIRECTOR — CONCREET ACTIEPLAN
    # ==========================================================

    print("\n" + "=" * 60)
    print("🚀 AI CONTENT DIRECTOR — CONCREET ACTIEPLAN")
    print("=" * 60)

    adviezen = []

    total_posts = len(df)

    hashtag_usage = {}

    for _, row in df.iterrows():
        tags = extract_hashtags(row["text"])
        for tag in tags:
            hashtag_usage[tag] = hashtag_usage.get(tag, 0) + 1

    if best_tags:
        primary_tag = best_tags[0]
        count = hashtag_usage.get(primary_tag, 0)

        current_freq = count / total_posts if total_posts > 0 else 0
        target_freq = min(current_freq * 1.3, 0.8)

        posts_out_of_10 = round(target_freq * 10)

        adviezen.append(
            f"Gebruik #{primary_tag} in {posts_out_of_10} van je volgende 10 posts ({int(target_freq*100)}%)."
        )

    adviezen.append(
        f"Gebruik exact {best_hashtags} hashtags per post."
    )

    adviezen.append(
        f"Schrijf captions van ±{avg_top_words} woorden."
    )

    if avg_top_emoji > 0:
        adviezen.append(
            f"Gebruik {avg_top_emoji} emoji per post."
        )

    adviezen.append(
        f"Post de komende 7 dagen rond {best_hour}:00."
    )

    print("\n🔥 WAT JE NU MOET DOEN:")
    print("-" * 60)

    for i, a in enumerate(adviezen, 1):
        print(f"{i}. {a}")


    return model, df


# ==============================
# ▶️ AANROEP
# ==============================

if combined.empty or len(combined) < 8:
    print("⚠️ Te weinig data voor AI")
    model = None
else:
    model, combined = train_personal_ai_model(combined)

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
