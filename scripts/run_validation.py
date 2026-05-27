"""Run the full 5-test validation scorecard on a signal.

Usage:
    uv run python scripts/run_validation.py --parquet data/processed/SYMBOL_TIMEFRAME.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from quant.backtest.engine import run_backtest
from quant.validation.scorecard import StrategyScorecard, ScorecardConfig


VERDICT_COLOR = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m"}
RESET = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",  required=True, help="Path to OHLCV parquet file")
    parser.add_argument("--capital",  type=float, default=10_000.0)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {Path(args.parquet).name}: {len(df):,} bars")

    # --- plug in your signal here ---
    signal = pd.Series(0, index=df.index, dtype=int)
    # signal = your_strategy.generate_signal(df)
    # --------------------------------

    result = run_backtest(df, signal, initial_capital=args.capital)
    card   = StrategyScorecard(config=ScorecardConfig()).evaluate(
        df=df, signal=signal, backtest_result=result
    )

    verdict = card["verdict"]
    color   = VERDICT_COLOR.get(verdict, "")
    print(f"\nVerdict: {color}{verdict.upper()}{RESET}  ({card['score']}/5 tests passed)\n")
    for name, passed in card["scores"].items():
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {name}")
    print()


if __name__ == "__main__":
    main()
