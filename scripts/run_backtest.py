"""Run a backtest and print metrics.

Usage:
    uv run python scripts/run_backtest.py --parquet data/processed/SYMBOL_TIMEFRAME.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from quant.backtest.engine import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",  required=True, help="Path to OHLCV parquet file")
    parser.add_argument("--capital",  type=float, default=10_000.0)
    parser.add_argument("--point-value", type=float, default=1.0, dest="point_value")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {Path(args.parquet).name}: {len(df):,} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # --- plug in your signal here ---
    import numpy as np
    signal = pd.Series(0, index=df.index, dtype=int)
    # signal = your_strategy.generate_signal(df)
    # --------------------------------

    result = run_backtest(df, signal, initial_capital=args.capital, point_value=args.point_value)
    m = result["metrics"]
    eq = result["equity_curve"]

    print(f"\n{'─'*40}")
    print(f"  Trades   : {m['n_trades']}")
    print(f"  PF       : {m['profit_factor']:.3f}")
    print(f"  Sharpe   : {m['sharpe']:.3f}")
    print(f"  Max DD   : {m['max_dd']*100:.1f}%")
    print(f"  Win Rate : {m['win_rate']*100:.1f}%")
    print(f"  End Eq   : {eq.iloc[-1]:,.2f}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
