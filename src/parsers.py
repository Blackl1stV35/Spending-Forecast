"""
KBank statement parsers.

Bank CSV structure (KBank savings account export):
  - Rows 0-8: metadata / header block
  - Row 9: column header (Date, Time, Description, Withdrawal, Deposit, Balance, Channel, Details)
  - Row 10+: transaction data

CC CSV structure (KBank credit card export):
  - Rows 0-4: card metadata
  - Row 5: column header  (Effective Date, Posting Date, Transfer Name, Transfer Amount)
  - Row 6+: transaction data
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def scan_person_files(person: str, data_dir: Path) -> dict:
    """Return dict with 'bank' and 'cc' lists of Path objects."""
    bank_dir = data_dir / person / "BankAccount"
    cc_dir = data_dir / person / "CreditCard"
    return {
        "bank": sorted(bank_dir.glob("*.csv")) if bank_dir.exists() else [],
        "cc": sorted(cc_dir.glob("*.csv")) if cc_dir.exists() else [],
    }


def _clean_amount(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def parse_bank_statement(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse a KBank savings account statement CSV.
    Returns a cleaned DataFrame or None on failure.
    """
    try:
        raw = pd.read_csv(
            filepath,
            header=None,
            encoding="utf-8-sig",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"[bank parser] Cannot read {filepath.name}: {exc}")
        return None

    # Locate the header row — look for the row that contains "Date" and "Withdrawal"
    header_row = None
    for i, row in raw.iterrows():
        row_str = " ".join(row.dropna().astype(str)).lower()
        if "withdrawal" in row_str and ("date" in row_str or "วันที่" in row_str):
            header_row = i
            break

    if header_row is None:
        # Fallback: try row index 9 which is typical for KBank exports
        header_row = 9

    try:
        df = pd.read_csv(
            filepath,
            header=header_row,
            encoding="utf-8-sig",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"[bank parser] Cannot read with header={header_row} in {filepath.name}: {exc}")
        return None

    if df.shape[1] < 5:
        return None

    # Map positional columns (KBank format has 13 cols)
    col_map = {
        0: "_skip",
        1: "Date",
        2: "Time",
        3: "Description",
        4: "Withdrawal",
        5: "_skip2",
        6: "Deposit",
        7: "_skip3",
        8: "Balance",
        9: "_skip4",
        10: "Channel",
        11: "_skip5",
        12: "Details",
    }
    df.columns = [col_map.get(i, f"_col{i}") for i in range(df.shape[1])]

    keep = [c for c in ["Date", "Time", "Description", "Withdrawal", "Deposit", "Balance", "Channel", "Details"] if c in df.columns]
    df = df[keep].copy()

    # Drop rows where Date looks like a header label or is NaN
    df = df.dropna(subset=["Date"])
    df = df[~df["Date"].astype(str).str.lower().str.contains(r"date|วันที่|nan", regex=True)]

    # Parse date — KBank uses DD-MM-YY
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Parse numeric columns
    for col in ["Withdrawal", "Deposit", "Balance"]:
        if col in df.columns:
            df[col] = _clean_amount(df[col])

    df["source"] = "bank"
    df["source_file"] = filepath.name

    return df.reset_index(drop=True)


def parse_credit_card(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse a KBank credit card statement CSV.
    Returns a cleaned DataFrame or None on failure.
    """
    try:
        lines = filepath.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        print(f"[cc parser] Cannot read {filepath.name}: {exc}")
        return None

    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if "Effective Date" in line or "effective date" in line.lower():
            header_idx = i
            break

    if header_idx is None:
        return None

    rows = []
    for line in lines[header_idx + 1 :]:
        line = line.strip()
        if not line:
            continue
        # Strip wrapping quotes and split
        parts = [p.strip().strip('"') for p in line.replace('","', "\t").split("\t")]
        if len(parts) < 4:
            parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(parts[:4])

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Eff_Date", "Post_Date", "Merchant", "Amount"])

    df["Amount"] = _clean_amount(df["Amount"])
    df["Date"] = pd.to_datetime(df["Eff_Date"], format="%d/%m/%Y", errors="coerce")

    # Keep only real purchases: positive amount, valid date, not payments / balance carry
    df = df.dropna(subset=["Date", "Amount"])
    df = df[df["Amount"] > 0]
    df = df[
        ~df["Merchant"]
        .str.upper()
        .str.contains(r"PAYMENT|PREVIOUS BALANCE|RETAIL DISPUTE", regex=True, na=False)
    ]

    df["source"] = "cc"
    df["source_file"] = filepath.name

    return df.reset_index(drop=True)


import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from io import BytesIO

from src.supabase_store import is_available, list_csv_files, download_csv
import streamlit as st

def load_person_data(
    person: str = "Kanokphan", 
    data_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and deduplicate bank + credit card transactions for a person.
    
    Loading priority (best to worst):
    1. Newly uploaded files in current session (via st.session_state)
    2. Previously uploaded files stored in Supabase (persistent)
    3. Local files in data_dir (fallback, mainly for local development)
    
    Returns: (bank_df, cc_df)
    """
    # --- 1. Collect all sources ---
    bank_sources: List = []
    cc_sources: List = []

    # A. Session uploads (highest priority - fresh uploads this session)
    bank_uploads = st.session_state.get(f"{person}_bank_uploads", [])
    cc_uploads = st.session_state.get(f"{person}_cc_uploads", [])

    bank_sources.extend(bank_uploads)
    cc_sources.extend(cc_uploads)

    # B. Load from Supabase (persistent storage)
    if is_available():
        try:
            # Bank files
            bank_files = list_csv_files(person, "bank")
            for meta in bank_files:
                content = download_csv(meta["storage_path"])
                if content:
                    fake_file = BytesIO(content)
                    fake_file.name = meta["filename"]
                    bank_sources.append(fake_file)

            # Credit card files
            cc_files = list_csv_files(person, "cc")
            for meta in cc_files:
                content = download_csv(meta["storage_path"])
                if content:
                    fake_file = BytesIO(content)
                    fake_file.name = meta["filename"]
                    cc_sources.append(fake_file)

            if bank_files or cc_files:
                st.sidebar.success(
                    f"✅ Loaded {len(bank_files)} bank + {len(cc_files)} CC files from Supabase"
                )
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load from Supabase: {str(e)}. Using local fallback.")

    # C. Fallback to local data/ directory (if provided and no data yet)
    if (not bank_sources and not cc_sources) and data_dir is not None:
        try:
            from src.parsers import scan_person_files   # keep your existing scanner

            files = scan_person_files(person, data_dir)
            
            # Convert local file paths to file-like objects (for consistent parsing)
            for f in files.get("bank", []):
                if f.exists():
                    bank_sources.append(f.open("rb"))
            for f in files.get("cc", []):
                if f.exists():
                    cc_sources.append(f.open("rb"))

            if files.get("bank") or files.get("cc"):
                st.sidebar.info("📁 Loaded data from local data/ folder")
        except Exception as e:
            st.sidebar.warning(f"Local data load failed: {str(e)}")

    # --- 2. Parse all sources ---
    bank_frames: List[pd.DataFrame] = []
    cc_frames: List[pd.DataFrame] = []

    # Import parsers here to avoid circular imports
    from src.parsers import parse_bank_statement, parse_credit_card

    for file_obj in bank_sources:
        try:
            df = parse_bank_statement(file_obj)
            if df is not None and len(df) > 0:
                bank_frames.append(df)
        except Exception as e:
            st.warning(f"Failed to parse bank file {getattr(file_obj, 'name', 'unknown')}: {e}")

    for file_obj in cc_sources:
        try:
            df = parse_credit_card(file_obj)
            if df is not None and len(df) > 0:
                cc_frames.append(df)
        except Exception as e:
            st.warning(f"Failed to parse CC file {getattr(file_obj, 'name', 'unknown')}: {e}")

    # --- 3. Combine + Deduplicate ---
    def combine_frames(frames: List[pd.DataFrame], sort_col: str = "Date") -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        
        combined = pd.concat(frames, ignore_index=True)
        
        # Smart key columns for deduplication (handles slight column name differences)
        key_cols = []
        for col in [sort_col, "Merchant", "Description", "Amount", "Withdrawal", "Deposit"]:
            if col in combined.columns:
                key_cols.append(col)
        
        # Fallback if no good keys found
        if not key_cols:
            key_cols = combined.columns.tolist()[:3]   # use first few columns

        # Deduplicate while keeping the first occurrence
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
        
        # Sort and clean
        if sort_col in combined.columns:
            combined = combined.sort_values(sort_col)
        
        return combined.reset_index(drop=True)

    bank_df = combine_frames(bank_frames, "Date")
    cc_df = combine_frames(cc_frames, "Date")

    total_rows = len(bank_df) + len(cc_df)
    if total_rows > 0:
        st.sidebar.info(f"Total transactions loaded: {total_rows:,} rows")
    else:
        st.sidebar.warning("No transaction data found.")

    return bank_df, cc_df


def load_from_uploads(bank_uploads: list, cc_uploads: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse uploaded file objects (from st.file_uploader).
    Saves them to temp paths and delegates to existing parsers.
    """
    import tempfile, os

    def _handle(uploads, parse_fn):
        frames = []
        for up in uploads:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(up.getvalue())
                tmp_path = Path(tmp.name)
            df = parse_fn(tmp_path)
            os.unlink(tmp_path)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("Date").reset_index(drop=True)

    bank_df = _handle(bank_uploads, parse_bank_statement)
    cc_df = _handle(cc_uploads, parse_credit_card)
    return bank_df, cc_df
