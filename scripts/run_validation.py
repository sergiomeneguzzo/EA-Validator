"""Run the 5-test validation scorecard on a strategy.

Usage:
    uv run python scripts/run_validation.py --parquet data/processed/EURUSD_H1.parquet --fast 20 --slow 50
"""

import argparse
from pathlib import Path

import pandas as pd

from quant.strategies import sma_cross
from quant.validation.scorecard import ScorecardConfig, StrategyScorecard

TESTS = ("purged_cv", "deflated_sr", "monte_carlo", "robustness", "walk_forward")
COLORS = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m"}
RESET = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="OHLCV parquet file")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--trials", type=int, default=10, help="parameter sets you tried before picking this one")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {Path(args.parquet).name}: {len(df):,} bars")

    strategy, params = sma_cross, {"fast": args.fast, "slow": args.slow}
    card = StrategyScorecard(ScorecardConfig()).evaluate("sma_cross", strategy, df, params, n_trials=args.trials)

    status = card["overall_status"]
    print(f"\nVerdict: {COLORS.get(status, '')}{status.upper()}{RESET}  ({card.get('n_tests_passed', 0)}/5 tests passed)\n")
    for name in TESTS:
        if name in card:
            print(f"  {'✓' if card[name]['pass'] else '✗'}  {name}")
    print()


if __name__ == "__main__":
    main()
