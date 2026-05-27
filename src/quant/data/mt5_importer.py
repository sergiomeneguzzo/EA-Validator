"""MT5 CSV importer — converts MetaTrader 5 exports to Parquet.

Supports two MT5 export formats:
  OHLCV: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
  Tick:  <DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>

Usage:
  df = load_symbol("SYMBOL")           # reads data/raw/SYMBOL_M1*.csv
  import_all()                          # converts all CSVs → data/processed/*.parquet
  df = load_parquet("SYMBOL", "H1")    # loads cached parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
log = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parents[3] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"

OHLCV_RENAME = {
    "<open>": "open", "<high>": "high", "<low>": "low", "<close>": "close",
    "<tickvol>": "volume", "<vol>": "vol", "<spread>": "spread",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "tickvol": "volume", "vol": "vol", "spread": "spread",
}
TICK_RENAME = {"<bid>": "bid", "<ask>": "ask", "<last>": "last",
               "<volume>": "volume", "<flags>": "flags", "bid": "bid", "ask": "ask"}


def _detect_format(path: Path) -> str:
    try:
        with open(path, "rb") as f: raw = f.read(512)
        if raw[:4] == b"book": return "binary"
        first_line = raw.decode("utf-8", errors="replace").splitlines()[0].lower()
    except Exception: return "unknown"
    if any(k in first_line for k in ("open", "<open>")): return "ohlcv"
    if any(k in first_line for k in ("bid", "<bid>", "ask", "<ask>")): return "tick"
    sep = "\t" if "\t" in first_line else ","
    return "ohlcv" if len(first_line.split(sep)) >= 7 else "tick"


def _sniff_separator(path: Path) -> str:
    with open(path, "rb") as f: sample = f.read(2048).decode("utf-8", errors="replace")
    return "\t" if sample.count("\t") > sample.count(",") else ","


def _parse_ohlcv(path: Path) -> pd.DataFrame:
    sep = _sniff_separator(path)
    raw = pd.read_csv(path, sep=sep, header=0, low_memory=False)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw.rename(columns={c: OHLCV_RENAME.get(c, c) for c in raw.columns}, inplace=True)

    date_col = next((c for c in raw.columns if "date" in c), None)
    time_col = next((c for c in raw.columns if "time" in c), None)
    if not date_col or not time_col:
        raise ValueError(f"Cannot find date/time columns in {path.name}")

    dt_str = raw[date_col].astype(str) + " " + raw[time_col].astype(str)
    raw["datetime"] = pd.to_datetime(dt_str, format="%Y.%m.%d %H:%M", utc=True, errors="coerce")
    mask = raw["datetime"].isna()
    if mask.any():
        raw.loc[mask, "datetime"] = pd.to_datetime(dt_str[mask], format="%Y.%m.%d %H:%M:%S",
                                                    utc=True, errors="coerce")
    raw = raw.set_index("datetime").sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume", "spread") if c in raw.columns]
    df   = raw[cols].copy()
    for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("volume", "spread"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "spread" not in df.columns: df["spread"] = 0.0
    if "volume" not in df.columns: df["volume"] = 0.0
    return df.dropna(subset=["open", "high", "low", "close"])


def _parse_tick(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    sep    = _sniff_separator(path)
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(path, sep=sep, header=0, low_memory=False, chunksize=chunksize)

    for chunk in reader:
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk.rename(columns={c: TICK_RENAME.get(c, c) for c in chunk.columns}, inplace=True)
        date_col = next((c for c in chunk.columns if "date" in c), None)
        time_col = next((c for c in chunk.columns if "time" in c), None)
        if not date_col or not time_col: continue

        dt_str = chunk[date_col].astype(str) + " " + chunk[time_col].astype(str)
        chunk["datetime"] = pd.to_datetime(dt_str, format="%Y.%m.%d %H:%M:%S.%f", utc=True, errors="coerce")
        mask = chunk["datetime"].isna()
        if mask.any():
            chunk.loc[mask, "datetime"] = pd.to_datetime(dt_str[mask], format="%Y.%m.%d %H:%M:%S",
                                                          utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["datetime"])
        if "bid" not in chunk.columns: continue

        chunk["bid"] = pd.to_numeric(chunk["bid"], errors="coerce")
        chunk["ask"] = pd.to_numeric(chunk.get("ask", chunk["bid"]), errors="coerce")
        chunk = chunk.dropna(subset=["bid"]).set_index("datetime").sort_index()

        m1 = chunk["bid"].resample("1min").agg(open="first", high="max", low="min", close="last").dropna()
        if "ask" in chunk.columns:
            m1["spread"] = (chunk["ask"] - chunk["bid"]).resample("1min").mean()
        else:
            m1["spread"] = 0.0
        m1["volume"] = chunk["bid"].resample("1min").count()
        chunks.append(m1)
        console.log(f"  chunk: {len(m1)} M1 bars")

    if not chunks:
        raise ValueError("No usable data found in tick file")

    df = pd.concat(chunks).sort_index()
    return df.groupby(df.index).agg(open="first", high="max", low="min", close="last",
                                     volume="sum", spread="mean")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    ohlc_flat   = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    spread_zero = df.get("spread", pd.Series(1, index=df.index)) == 0
    df = df[~(ohlc_flat & spread_zero)].copy()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna(subset=["open", "high", "low", "close"])


def load_symbol(symbol: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and clean one symbol's M1 OHLCV data from raw CSV."""
    candidates = list(raw_dir.glob(f"{symbol}_M1*.csv")) + list(raw_dir.glob(f"{symbol}_M1*.CSV"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {symbol} in {raw_dir}")
    path = candidates[0]
    fmt  = _detect_format(path)
    if fmt == "binary":
        raise ValueError(f"{path.name} is a macOS alias file. Re-export from MT5.")
    df = _parse_tick(path) if fmt == "tick" else _parse_ohlcv(path)
    df = _clean(df)
    console.log(f"  {symbol}: {len(df):,} M1 bars  [{df.index[0]} → {df.index[-1]}]")
    return df


def import_all(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert all symbol CSVs to parquet and return dict."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for sym in (symbols or [p.stem.split("_M1")[0] for p in raw_dir.glob("*_M1*.csv")]):
        try:
            console.log(f"[bold]Importing[/] {sym}...")
            df  = load_symbol(sym, raw_dir)
            out = processed_dir / f"{sym}_M1.parquet"
            df.to_parquet(out)
            console.log(f"  → saved {out.name}")
            result[sym] = df
        except FileNotFoundError:
            console.log(f"[yellow]SKIP[/] {sym}: no CSV found")
        except ValueError as exc:
            console.log(f"[red]ERROR[/] {sym}: {exc}")
    return result


def load_parquet(symbol: str, timeframe: str = "M1", processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load a processed parquet file."""
    path = processed_dir / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run import_all() first")
    return pd.read_parquet(path)
