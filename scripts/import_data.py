"""Convert MT5 CSV exports to Parquet.

Usage:
    uv run python scripts/import_data.py
    uv run python scripts/import_data.py --symbol SYMBOL --timeframe H1
"""

import argparse
from pathlib import Path

from quant.data.mt5_importer import import_all, load_symbol


def main() -> None:
    parser = argparse.ArgumentParser(description="Import MT5 CSV → Parquet")
    parser.add_argument("--symbol",    default=None, help="Single symbol (e.g. EURUSD)")
    parser.add_argument("--timeframe", default=None, help="Single timeframe (e.g. H1)")
    parser.add_argument("--raw-dir",   default="data/raw",       dest="raw_dir")
    parser.add_argument("--out-dir",   default="data/processed", dest="out_dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.symbol and args.timeframe:
        df = load_symbol(args.symbol, args.timeframe, raw_dir=raw_dir)
        out = out_dir / f"{args.symbol}_{args.timeframe}.parquet"
        df.to_parquet(out)
        print(f"Saved {len(df):,} rows → {out}")
    else:
        results = import_all(raw_dir=raw_dir, out_dir=out_dir)
        for path, n in results.items():
            print(f"{path.name}: {n:,} rows")


if __name__ == "__main__":
    main()
