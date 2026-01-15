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
