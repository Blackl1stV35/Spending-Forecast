"""
Keyword-based transaction categoriser.

Each transaction's text (merchant name or description + details) is matched
against keyword dictionaries in config.py.  First match wins.

──────────────────────────────────────────────────────────────────────────────
TRANSFER FILTER DESIGN
──────────────────────────────────────────────────────────────────────────────
KBank QR / PromptPay bank payments ALL arrive as "Paid for Ref XXXX <merchant>".
The previous filter matched the string "paid for ref" and wiped out 101 real
merchant transactions (฿104k).  The corrected rules below are SURGICAL:

Only exclude a bank row when it is a genuine double-count or non-spending flow:
  1. "kbank card"      — bank pays off credit card bill (already in CC statement)
  2. "kasikorn securi" — investment brokerage transfer
  3. "ksecurities"     — investment brokerage variant

Everything else — PromptPay payments, Transfer Withdrawal to individuals,
"Paid for Ref" merchant QR payments — is REAL spending and must be kept.

──────────────────────────────────────────────────────────────────────────────
OTHER-BUCKET FIX
──────────────────────────────────────────────────────────────────────────────
Two expansion layers applied in sequence after config.py rules:

  Layer 1 – BANK_KEYWORD_EXTRA:  patterns extracted from actual KBank Detail
             strings.  Covers Thai merchant names, SCB มณี SHOP wrappers,
             payment intermediaries, utilities, EV charging, etc.

  Layer 2 – MERCHANT_OVERRIDES:  applied only to rows still in "Other".
             Covers CC merchant names and common international brands.
"""

from __future__ import annotations

import pandas as pd

from src.config import BANK_CATEGORIES, CC_CATEGORIES

# ─────────────────────────────────────────────────────────────────────────────
# EXCLUSION: true double-count + investment flows only
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_PATTERNS: list[str] = [
    "kbank card",           # bank CC bill payment — already in CC statement
    "kasikorn securitie",   # investment brokerage
    "ksecurities",
    "หลักทรัพย์",           # Thai: "securities"
]

# Person-to-person transfers: kept in spending, categorised as Family/Personal
P2P_PATTERNS: list[str] = [
    "to promptpay",
    "to x",
    "to ttb",
]


def _should_exclude(merchant: str, description: str = "") -> bool:
    """True only for genuine double-count or investment rows."""
    text = (str(merchant) + " " + str(description)).lower()
    return any(p in text for p in EXCLUDE_PATTERNS)


def _is_p2p_transfer(merchant: str) -> bool:
    m = str(merchant).lower()
    return any(p in m for p in P2P_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — EXTENDED BANK KEYWORD MAP
# Patterns extracted directly from real KBank Detail strings.
# ─────────────────────────────────────────────────────────────────────────────

BANK_KEYWORD_EXTRA: dict[str, list[str]] = {
    "Utilities": [
        "mea",                  # Metropolitan Electricity Authority
        "pea",                  # Provincial Electricity Authority
        "mwa",                  # Metropolitan Waterworks
        "pwa",                  # Provincial Waterworks
        "true money",           # TrueMoney wallet / utility top-up
        "advance mpay",         # AIS mPAY
        "advanced mpay",
        "ntt data",             # payment gateway for utility bills
        "การไฟฟ้า",
        "ประปา",
        "ค่าน้ำ",
        "ค่าไฟ",
    ],
    "Food & Dining": [
        "โกเฮง",               # Go Heng chicken rice
        "go heng",
        "ข้าวมันไก่",
        "สุกี้ตี๋น้อย",         # Suki Teenage (sukiyaki chain)
        "โกกิเนมเซ",           # Korean restaurant
        "เปียงยาง",             # Pyongyang Korean restaurant
        "ครัวช้อนใหญ่",
        "ปภาภัทร ไอศกรีม",
        "minor dq",             # Dairy Queen
        "กันปาย",
        "jenkongklai",
        "gourmet galaxy",
        "ร้านถุงเงิน",
        "toppop snow",
        "รอล่า",
        "วันเฮง",
        "ซุปผักลาว",
        "ร้านเวนดิ้ง",          # vending machine (snacks)
        "เวนดิ้งบายบุญเติม",
        "นุ้ยการค้า",
        "scb มณี shop",         # SCB QR wrapper — catches all sub-merchants
        "scm mani",
        "สยาม ซีนีเพล็กซ์",     # Siam Cineplex canteen / food court (also entertainment)
    ],
    "Beverages & Cafe": [
        "กาแฟพันธุ์ไทย",
        "ชอบชา",
        "โนบิชา",
        " tea",
        "boba",
        "ชานม",
    ],
    "Travel": [
        "youtrip",              # forex prepaid card top-up
        "ซุปเปอร์ริช",          # SuperRich currency exchange
        "superrich",
        "สนามบิน",
        "ท่าอากาศยาน",
        "smartcarpark",
        "airport",
    ],
    "Transport": [
        "car park",
        "parking",
        "ที่จอดรถ",
        "north park propert",   # apartment/carpark recurring fee
        "green propulsion",     # EV charging
        "กรีน โพรพัลชั่น",
    ],
    "Home & Services": [
        "โปรไทย คลีนนิ่ง",
        "cleaning",
        "ทำความสะอาด",
        "แม่บ้าน",
    ],
    "Entertainment": [
        "สยาม ซีนีเพล็กซ์",
        "cineplex",
        "major 1007",
        "sf cinema",
        "karaoke",
        "bowling",
    ],
    "Shopping & Fashion": [
        "awr-ast",              # Asset World Retail
        "asset world",
        "แอสเสท เวิรด์",
        "central pattana",
    ],
    "Government & Fees": [
        "สำนักงานเขต",
        "กรมสรรพากร",
        "transport departm",
        "กรมขนส่ง",
    ],
    "Healthcare": [
        "เภสัช",
        "ร้านยา",
        "คมสัน เภสัช",
        "dental",
        "optical",
        "eye care",
    ],
    "Family / Personal": [
        "athipat suri",
        "nurulaiman",
        "mandeep singh",
        "theint theint",
        "itsarapa",
        "nuttanan leart",
        "rapeerat",
        "nathorn",
        "arpaporn",
        "warisara",
        "amonsak",
        "punnatut",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — MERCHANT OVERRIDE DICT
# Applied only when category is still "Other" after layers 0 and 1.
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_OVERRIDES: dict[str, str] = {
    # Fuel
    "shell": "Fuel",
    "caltex": "Fuel",
    "esso": "Fuel",
    "pt gas": "Fuel",
    "irpc": "Fuel",
    # Coffee / cafe
    "starbucks": "Beverages & Cafe",
    "cafe amazon": "Beverages & Cafe",
    "black canyon": "Beverages & Cafe",
    "inthanin": "Beverages & Cafe",
    "doi chaang": "Beverages & Cafe",
    "the coffee club": "Beverages & Cafe",
    "coffee world": "Beverages & Cafe",
    "swensen": "Beverages & Cafe",
    # Fast food / dining
    "mcdonald": "Food & Dining",
    "burger king": "Food & Dining",
    "kfc": "Food & Dining",
    "pizza hut": "Food & Dining",
    "pizza company": "Food & Dining",
    "domino": "Food & Dining",
    "subway": "Food & Dining",
    "sizzler": "Food & Dining",
    "sukishi": "Food & Dining",
    "fuji": "Food & Dining",
    "zen restaurant": "Food & Dining",
    "dairy queen": "Food & Dining",
    # Grocery
    "villa market": "Groceries",
    "gourmet market": "Groceries",
    "rimping": "Groceries",
    "foodland": "Groceries",
    "makro": "Groceries",
    "big c": "Groceries",
    "bigc": "Groceries",
    "maxvalu": "Groceries",
    "fresh market": "Groceries",
    # Pharmacy / health
    "watsons": "Healthcare",
    "boots": "Healthcare",
    "guardian": "Healthcare",
    # Transport
    "taxi": "Transport",
    "mrt": "Transport",
    "bts": "Transport",
    "airport rail": "Transport",
    "grab": "Transport",
    "bolt": "Transport",
    "rapido": "Transport",
    # Tech / digital
    "netflix": "Tech & Digital",
    "spotify": "Tech & Digital",
    "youtube": "Tech & Digital",
    "apple.com": "Tech & Digital",
    "apple store": "Tech & Digital",
    "google play": "Tech & Digital",
    "google one": "Tech & Digital",
    "line pay": "Tech & Digital",
    "runpod": "Tech & Digital",
    "openai": "Tech & Digital",
    "anthropic": "Tech & Digital",
    "aws": "Tech & Digital",
    "adobe": "Tech & Digital",
    "dropbox": "Tech & Digital",
    "github": "Tech & Digital",
    # Shopping
    "shopee": "Shopping & Fashion",
    "lazada": "Shopping & Fashion",
    "(for shopee)": "Shopping & Fashion",
    "amz": "Shopping & Fashion",
    "amazon": "Shopping & Fashion",
    "central": "Shopping & Fashion",
    "robinson": "Shopping & Fashion",
    "zara": "Shopping & Fashion",
    "h&m": "Shopping & Fashion",
    "uniqlo": "Shopping & Fashion",
    "adidas": "Shopping & Fashion",
    "nike": "Shopping & Fashion",
    # Travel
    "superrich": "Travel",
    "youtrip": "Travel",
    "agoda": "Travel",
    "booking.com": "Travel",
    "airbnb": "Travel",
    "airasia": "Travel",
    "thai airways": "Travel",
    "nok air": "Travel",
}


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _match(text: str, rules: dict) -> str:
    t = str(text).lower()
    for category, keywords in rules.items():
        if any(kw.lower() in t for kw in keywords):
            return category
    return "Other"


def _match_extra(text: str) -> str:
    t = str(text).lower()
    for category, keywords in BANK_KEYWORD_EXTRA.items():
        if any(kw.lower() in t for kw in keywords):
            return category
    return "Other"


def _apply_merchant_overrides(category: str, merchant: str) -> str:
    if category != "Other":
        return category
    m = str(merchant).lower()
    for keyword, override_cat in MERCHANT_OVERRIDES.items():
        if keyword in m:
            return override_cat
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def categorise_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Category column to a bank withdrawal DataFrame.

    Matching priority (first match wins):
      0. BANK_CATEGORIES (config.py)
      1. BANK_KEYWORD_EXTRA (Thai names, intermediaries, utilities)
      2. MERCHANT_OVERRIDES (last-resort CC-style merchants)
      3. P2P heuristic → 'Family / Personal'
    """
    if df.empty:
        return df
    df = df.copy()
    text_series = (
        df.get("Details", pd.Series([""] * len(df), dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Description", pd.Series([""] * len(df), dtype=str)).fillna("").astype(str)
    )
    merchant_series = df.get(
        "Details", pd.Series([""] * len(df), dtype=str)
    ).fillna("").astype(str)

    def _classify(row_text: str, merchant: str) -> str:
        cat = _match(row_text, BANK_CATEGORIES)           # layer 0
        if cat != "Other":
            return cat
        cat = _match_extra(row_text)                       # layer 1
        if cat != "Other":
            return cat
        cat = _apply_merchant_overrides(cat, merchant)     # layer 2
        if cat != "Other":
            return cat
        if _is_p2p_transfer(merchant):                     # layer 3
            return "Family / Personal"
        return "Other"

    df["Category"] = [
        _classify(text_series.iloc[i], merchant_series.iloc[i])
        for i in range(len(df))
    ]
    return df


def categorise_cc(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Category column to a CC DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    merchant_col = df["Merchant"].fillna("").astype(str)

    def _classify_cc(merchant: str) -> str:
        cat = _match(merchant, CC_CATEGORIES)
        if cat != "Other":
            return cat
        cat = _match_extra(merchant)
        if cat != "Other":
            return cat
        return _apply_merchant_overrides(cat, merchant)

    df["Category"] = merchant_col.apply(_classify_cc)
    return df


def get_spending_df(bank_df: pd.DataFrame, cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unified, clean spending DataFrame.

    Exclusion logic (surgical):
      - Only CC bill payments and investment transfers are excluded.
      - All 'Paid for Ref' QR merchant payments are KEPT.
      - PromptPay and Transfer Withdrawal outflows are KEPT.

    Output columns:
        Date, Amount, Merchant, Category, source, YearMonth, DayOfWeek
    """
    frames: list[pd.DataFrame] = []

    if not bank_df.empty:
        b = categorise_bank(bank_df)
        mask = b["Withdrawal"].notna() & (b["Withdrawal"] > 0)
        out = b[mask].copy()
        out["Amount"] = out["Withdrawal"]
        out["Merchant"] = out.apply(
            lambda r: (
                str(r.get("Details", "")).strip()
                or str(r.get("Description", "")).strip()
            ),
            axis=1,
        )
        frames.append(out[["Date", "Amount", "Merchant", "Category", "source"]].copy())

    if not cc_df.empty:
        c = categorise_cc(cc_df)
        spend = c[c["Amount"] > 0].copy()
        frames.append(spend[["Date", "Amount", "Merchant", "Category", "source"]].copy())

    if not frames:
        return pd.DataFrame(
            columns=["Date", "Amount", "Merchant", "Category", "source", "YearMonth", "DayOfWeek"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    # Surgical exclusion
    exclude_mask = combined["Merchant"].apply(lambda m: _should_exclude(m))
    n_excluded = exclude_mask.sum()
    if n_excluded > 0:
        excluded_amt = combined.loc[exclude_mask, "Amount"].sum()
        print(
            f"[categoriser] Excluded {n_excluded} rows "
            f"(฿{excluded_amt:,.0f}) — CC payments / investment."
        )
    combined = combined[~exclude_mask].copy()

    combined["YearMonth"] = combined["Date"].dt.to_period("M").dt.to_timestamp()
    combined["DayOfWeek"] = combined["Date"].dt.day_name()

    return combined.sort_values("Date").reset_index(drop=True)
