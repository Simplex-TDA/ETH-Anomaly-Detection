"""Progress tracker for run_ranking_mode_a_only.py (Mode A daily edge-based
ranking, FINAL-corrected data, 2020-2025).

Coarse progress: reads each year's weekly daily-ranking parquet files
(lightweight -- only the filter/ranking_metric/date columns), counts
distinct (filter, ranking_metric, date) triples actually computed, and
compares against the expected total for that year (filters-per-metric x
days-in-year, matching run_ranking_mode_a_only.py's METRIC_FILTER_GROUPS
exactly).

Usage:
    python3 scripts/monitoring/check_ranking_progress.py
"""
import os
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/monitoring/)
os.chdir(ROOT)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAYS_IN_YEAR = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
LOG_FILE = Path("logs/run_ranking_mode_a_only.log")

ALL_FILTERS = [
    "contract_txs_ETH_only", "simple_txs_ETH_only", "contract_factory_ETH_only",
    "contract_nonFactory_ETH_only", "contract_highInput_ETH_only", "contract_mediumInput_ETH_only",
    "contract_txs_ALL", "simple_txs_ALL", "contract_factory_ALL",
    "contract_nonFactory_ALL", "contract_highInput_ALL", "contract_mediumInput_ALL",
    "contract_txs_ERC20_only", "simple_txs_ERC20_only",
]
CONTRACT_FILTER_NAMES = [f for f in ALL_FILTERS if f.startswith("contract_")]
ERC20_ONLY_FILTER_NAMES = [f for f in ALL_FILTERS if f.endswith("_ERC20_only")]

METRIC_FILTER_GROUPS = {
    "tx_count": list(ALL_FILTERS),
    "tx_value": [f for f in ALL_FILTERS if f not in CONTRACT_FILTER_NAMES],
    "total_gas_fees": [f for f in ALL_FILTERS if f not in ERC20_ONLY_FILTER_NAMES],
}
FILTERS_PER_DAY = sum(len(v) for v in METRIC_FILTER_GROUPS.values())  # per day, summed across metrics


def triples_done(year):
    daily_dir = Path(f"data/ranking/{year}/daily")
    if not daily_dir.exists():
        return 0
    seen = set()
    for f in daily_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["filter", "ranking_metric", "date"])
        except Exception:
            continue
        seen.update(zip(df["filter"], df["ranking_metric"], pd.to_datetime(df["date"])))
    return len(seen)


def current_position():
    if not LOG_FILE.exists():
        return None, None
    text = LOG_FILE.read_text(errors="replace")
    years = re.findall(r"YEAR (\d{4})", text)
    metrics = re.findall(r"metric=(\S+)", text)
    return (years[-1] if years else None, metrics[-1] if metrics else None)


def main():
    print("=== Mode A ranking progress (FINAL-corrected data, 2020-2025) ===\n")
    grand_total = 0
    grand_done = 0
    for year in YEARS:
        total = FILTERS_PER_DAY * DAYS_IN_YEAR[year]
        done = triples_done(year)
        grand_total += total
        grand_done += done
        pct = 100 * done / total if total else 0
        print(f"  {year}: {done:>6,} / {total:,} (filter, metric, date) triples ({pct:5.1f}%)")

    pct = 100 * grand_done / grand_total if grand_total else 0
    print(f"\nOverall: {grand_done:,} / {grand_total:,} ({pct:.1f}%)")

    year, metric = current_position()
    if year:
        print(f"\nCurrently on: YEAR {year}, metric={metric or '?'}")

    if LOG_FILE.exists():
        age_s = time.time() - LOG_FILE.stat().st_mtime
        print(f"\nLog last updated {age_s:.0f}s ago", end="")
        if age_s > 300:
            print("  [!] no log activity in 5+ min -- check the process is alive")
        else:
            print()


if __name__ == "__main__":
    main()
