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


def load_person_data(person: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and deduplicate all bank + CC files for a person.
    Returns (bank_df, cc_df).
    """
    files = scan_person_files(person, data_dir)

    bank_frames = [parse_bank_statement(f) for f in files["bank"]]
    bank_frames = [f for f in bank_frames if f is not None and len(f) > 0]

    cc_frames = [parse_credit_card(f) for f in files["cc"]]
    cc_frames = [f for f in cc_frames if f is not None and len(f) > 0]

    def combine(frames: list, sort_col: str) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        # Deduplicate on key columns
        key_cols = [c for c in [sort_col, "Merchant" if "Merchant" in combined.columns else "Description", "Amount" if "Amount" in combined.columns else "Withdrawal"] if c in combined.columns]
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
        return combined.sort_values(sort_col).reset_index(drop=True)

    bank_df = combine(bank_frames, "Date")
    cc_df = combine(cc_frames, "Date")

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
                tmp.write(up.read())
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
