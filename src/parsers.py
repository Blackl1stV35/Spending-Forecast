"""
KBank statement parsers.

Bank CSV structure (KBank savings account export):
  - Rows 0-8: metadata / header block
  - Row 9: column header (Date, Time, Description, Withdrawal, Deposit, Balance, Channel, Details)
  - Row 10+: transaction data

CC CSV structure (KBank credit card export):
  - Rows 0-4: card metadata
  - Row 5: column header (Effective Date, Posting Date, Transfer Name, Transfer Amount)
  - Row 6+: transaction data

Loading priority for load_person_data():
  1. Local data/<person>/BankAccount/*.csv   (fastest, works offline)
  2. Supabase Storage csv-uploads bucket      (cloud-persisted uploads)

Both sources are combined and deduplicated so uploading a file that already
exists locally does not produce duplicate rows.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_amount(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def _bytes_to_tmp(content: bytes, suffix: str = ".csv") -> Path:
    """Write bytes to a NamedTemporaryFile and return its Path."""
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return Path(tmp_name)


# ─────────────────────────────────────────────────────────────────────────────
# File scanners
# ─────────────────────────────────────────────────────────────────────────────

def scan_person_files(person: str, data_dir: Path) -> dict:
    """Return dict with 'bank' and 'cc' lists of local Path objects."""
    bank_dir = data_dir / person / "BankAccount"
    cc_dir   = data_dir / person / "CreditCard"
    return {
        "bank": sorted(bank_dir.glob("*.csv")) if bank_dir.exists() else [],
        "cc":   sorted(cc_dir.glob("*.csv"))   if cc_dir.exists()   else [],
    }


def _fetch_supabase_files(person: str) -> dict[str, list[bytes]]:
    """
    Download all CSV files stored in Supabase Storage for a person.

    Returns {"bank": [<bytes>, ...], "cc": [<bytes>, ...]}
    Returns empty lists (not None) when Supabase is unavailable or has no files.
    """
    result: dict[str, list[bytes]] = {"bank": [], "cc": []}

    try:
        from src.supabase_store import list_csv_files, download_csv, is_available
        if not is_available():
            return result

        for source_type in ("bank", "cc"):
            file_records = list_csv_files(person, source_type=source_type)
            for record in file_records:
                storage_path = record.get("storage_path", "")
                if not storage_path:
                    continue
                content = download_csv(storage_path)
                if content:
                    result[source_type].append(content)
    except Exception:
        pass  # degrade gracefully — local files will still load

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core parsers (accept Path only — bytes callers use _bytes_to_tmp first)
# ─────────────────────────────────────────────────────────────────────────────

def parse_bank_statement(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse a KBank savings account statement CSV from a file path.
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

    # Locate header row — contains "withdrawal" and "date" / "วันที่"
    header_row = None
    for i, row in raw.iterrows():
        row_str = " ".join(row.dropna().astype(str)).lower()
        if "withdrawal" in row_str and ("date" in row_str or "วันที่" in row_str):
            header_row = i
            break
    if header_row is None:
        header_row = 9  # KBank default

    try:
        df = pd.read_csv(
            filepath,
            header=header_row,
            encoding="utf-8-sig",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"[bank parser] header={header_row} failed in {filepath.name}: {exc}")
        return None

    if df.shape[1] < 5:
        return None

    col_map = {
        0: "_skip", 1: "Date", 2: "Time", 3: "Description",
        4: "Withdrawal", 5: "_skip2", 6: "Deposit", 7: "_skip3",
        8: "Balance", 9: "_skip4", 10: "Channel", 11: "_skip5", 12: "Details",
    }
    df.columns = [col_map.get(i, f"_col{i}") for i in range(df.shape[1])]

    keep = [c for c in ["Date","Time","Description","Withdrawal","Deposit","Balance","Channel","Details"]
            if c in df.columns]
    df = df[keep].copy()

    df = df.dropna(subset=["Date"])
    df = df[~df["Date"].astype(str).str.lower().str.contains(r"date|วันที่|nan", regex=True)]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y", errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Withdrawal", "Deposit", "Balance"]:
        if col in df.columns:
            df[col] = _clean_amount(df[col])

    df["source"]      = "bank"
    df["source_file"] = filepath.name

    return df.reset_index(drop=True)


def parse_credit_card(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse a KBank credit card statement CSV from a file path.
    Returns a cleaned DataFrame or None on failure.
    """
    try:
        lines = filepath.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        print(f"[cc parser] Cannot read {filepath.name}: {exc}")
        return None

    header_idx = None
    for i, line in enumerate(lines):
        if "Effective Date" in line or "effective date" in line.lower():
            header_idx = i
            break
    if header_idx is None:
        return None

    rows = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.replace('","', "\t").split("\t")]
        if len(parts) < 4:
            parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(parts[:4])

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Eff_Date", "Post_Date", "Merchant", "Amount"])
    df["Amount"] = _clean_amount(df["Amount"])
    df["Date"]   = pd.to_datetime(df["Eff_Date"], format="%d/%m/%Y", errors="coerce")

    df = df.dropna(subset=["Date", "Amount"])
    df = df[df["Amount"] > 0]
    df = df[
        ~df["Merchant"]
        .str.upper()
        .str.contains(r"PAYMENT|PREVIOUS BALANCE|RETAIL DISPUTE", regex=True, na=False)
    ]

    df["source"]      = "cc"
    df["source_file"] = filepath.name

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bytes-level parse wrappers (used by Supabase + st.file_uploader paths)
# ─────────────────────────────────────────────────────────────────────────────

def parse_bank_bytes(content: bytes, filename: str = "upload.csv") -> Optional[pd.DataFrame]:
    """Parse bank CSV from raw bytes.  Cleans up the temp file automatically."""
    tmp = _bytes_to_tmp(content)
    try:
        df = parse_bank_statement(tmp)
        if df is not None:
            df["source_file"] = filename
        return df
    finally:
        tmp.unlink(missing_ok=True)


def parse_credit_card_bytes(content: bytes, filename: str = "upload.csv") -> Optional[pd.DataFrame]:
    """Parse CC CSV from raw bytes.  Cleans up the temp file automatically."""
    tmp = _bytes_to_tmp(content)
    try:
        df = parse_credit_card(tmp)
        if df is not None:
            df["source_file"] = filename
        return df
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication helper
# ─────────────────────────────────────────────────────────────────────────────

def _combine(frames: list[pd.DataFrame], sort_col: str) -> pd.DataFrame:
    """Concatenate, deduplicate, and sort a list of DataFrames."""
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    amount_col = "Amount" if "Amount" in combined.columns else "Withdrawal"
    merchant_col = "Merchant" if "Merchant" in combined.columns else "Description"
    key_cols = [c for c in [sort_col, merchant_col, amount_col] if c in combined.columns]
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    return combined.sort_values(sort_col).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_person_data(
    person: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all bank + CC data for a person from two sources in priority order:

      1. Local  data/<person>/BankAccount/*.csv  (instant, works offline)
      2. Supabase Storage csv-uploads bucket     (cloud-persisted uploads)

    Both sources are merged and deduplicated.  If Supabase is not configured
    or has no files for this person, only local files are returned.

    Returns (bank_df, cc_df).
    """
    # ── Source 1: local disk ──────────────────────────────────────────────────
    local = scan_person_files(person, data_dir)

    bank_frames: list[pd.DataFrame] = [
        df for f in local["bank"]
        if (df := parse_bank_statement(f)) is not None and len(df) > 0
    ]
    cc_frames: list[pd.DataFrame] = [
        df for f in local["cc"]
        if (df := parse_credit_card(f)) is not None and len(df) > 0
    ]

    # Track which filenames are already loaded (for dedup at the frame level)
    loaded_bank_names: set[str] = {f.name for f in local["bank"]}
    loaded_cc_names:   set[str] = {f.name for f in local["cc"]}

    # ── Source 2: Supabase Storage ────────────────────────────────────────────
    supabase_files = _fetch_supabase_files(person)

    for content in supabase_files["bank"]:
        # Derive a filename from the content hash to use as source_file label
        import hashlib
        fname = "supabase_bank_" + hashlib.md5(content[:512]).hexdigest()[:8] + ".csv"
        if fname in loaded_bank_names:
            continue  # already loaded from local disk
        df = parse_bank_bytes(content, filename=fname)
        if df is not None and len(df) > 0:
            bank_frames.append(df)
            loaded_bank_names.add(fname)

    for content in supabase_files["cc"]:
        import hashlib
        fname = "supabase_cc_" + hashlib.md5(content[:512]).hexdigest()[:8] + ".csv"
        if fname in loaded_cc_names:
            continue
        df = parse_credit_card_bytes(content, filename=fname)
        if df is not None and len(df) > 0:
            cc_frames.append(df)
            loaded_cc_names.add(fname)

    bank_df = _combine(bank_frames, "Date")
    cc_df   = _combine(cc_frames,   "Date")

    return bank_df, cc_df


def load_from_uploads(
    bank_uploads: list,
    cc_uploads: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse uploaded file objects from st.file_uploader.

    Each upload object must have a .read() method returning bytes and
    a .name attribute.  Delegates to the bytes-level parse wrappers.
    """
    def _handle(uploads: list, parse_fn) -> pd.DataFrame:
        frames = []
        for up in uploads:
            try:
                content = up.read()
                filename = getattr(up, "name", "upload.csv")
                df = parse_fn(content, filename=filename)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception as exc:
                print(f"[upload parser] Failed to parse {getattr(up, 'name', '?')}: {exc}")
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("Date").reset_index(drop=True)

    bank_df = _handle(bank_uploads, parse_bank_bytes)
    cc_df   = _handle(cc_uploads,   parse_credit_card_bytes)
    return bank_df, cc_df
