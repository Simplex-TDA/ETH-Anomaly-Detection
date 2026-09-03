"""Driver: daily ranking (Mode A, edge-based) only, for 2020-2025, against
the FINAL-corrected refetched raw data.

Mode B (centrality) intentionally skipped -- STATUS.md's prior finding
("Centrality (Mode B) mostly reproduces edge-based node selection (Mode
A)... its value is more likely as a per-node feature than as an alternative
node-selection mechanism") already established it as non-decisive, and no
current or planned TDA diagram build (V1's 8 layers or V2's 14 layers)
consumes Mode B output -- every diagram build reads Mode A's edge-based
daily ranking via `ranking_metric` (tx_count/tx_value/total_gas_fees).
Skipping Mode B saves a real, substantial amount of compute on an already
multi-day rebuild, with nothing on the critical path depending on it.

Same 14 filters and 3 metrics as run_ranking_all_years.py, identical
degenerate-combination exclusions (tx_value on contract_* filters,
total_gas_fees on *_ERC20_only filters).

Resumable by construction (run_daily_ranking tracks completed
(filter, ranking_metric, date) triples in the daily parquet files and
skips anything already there).

    python3 scripts/runs/run_ranking_mode_a_only.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/runs/)
os.chdir(ROOT)
sys.path.insert(0, "functions")

import pandas as pd
from ranking_functions import check_directories, run_daily_ranking

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TOP_N = 2000
DEAD_ADDRESSES = [None, "\\N"]

# Same 14 filters as run_ranking_all_years.py / 2_ranking.ipynb.
ALL_FILTERS = {
    "contract_txs_ETH_only": lambda d: (d["tx_value"] == 0) & (d["erc20"] == "ETH"),
    "simple_txs_ETH_only":   lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"),
    "contract_factory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["erc20"] == "ETH"),
    "contract_nonFactory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"),
    "contract_highInput_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"),
    "contract_mediumInput_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"),
    "contract_txs_ALL": lambda d: (d["tx_value"] == 0),
    "simple_txs_ALL":   lambda d: (d["tx_value"] >  0),
    "contract_factory_ALL": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100),
    "contract_nonFactory_ALL": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100),
    "contract_highInput_ALL": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500),
    "contract_mediumInput_ALL": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500),
    "contract_txs_ERC20_only": lambda d: (d["tx_value"] == 0) & (d["erc20"] != "ETH"),
    "simple_txs_ERC20_only":   lambda d: (d["tx_value"] >  0) & (d["erc20"] != "ETH"),
}

CONTRACT_FILTER_NAMES = [f for f in ALL_FILTERS if f.startswith("contract_")]
ERC20_ONLY_FILTER_NAMES = [f for f in ALL_FILTERS if f.endswith("_ERC20_only")]

METRIC_FILTER_GROUPS = [
    ("tx_count", list(ALL_FILTERS)),
    ("tx_value", [f for f in ALL_FILTERS if f not in CONTRACT_FILTER_NAMES]),
    ("total_gas_fees", [f for f in ALL_FILTERS if f not in ERC20_ONLY_FILTER_NAMES]),
]
KEEP_COLS = ["date", "from_addr", "to_addr", "tx_count", "tx_value", "total_gas_fees"]

for metric, filter_names in METRIC_FILTER_GROUPS:
    print(f"{metric}: {len(filter_names)} filters -> {filter_names}")

for year in YEARS:
    t0 = time.time()
    print(f"\n{'='*80}\nYEAR {year}\n{'='*80}", flush=True)

    eth_dir = Path(f"data/{year}/eth_tx_value_output/weekly")
    erc20_dir = Path(f"data/{year}/erc20_tx_value_output/weekly")
    ranking_dir = Path(f"data/ranking/{year}")
    daily_dir = ranking_dir / "daily"
    index_file = ranking_dir / "source_index.json"
    daily_dir.mkdir(parents=True, exist_ok=True)

    if not check_directories(eth_dir, erc20_dir):
        print(f"[{year}] SKIP -- missing source data.", flush=True)
        continue

    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31")

    for metric, filter_names in METRIC_FILTER_GROUPS:
        filters_subset = {name: ALL_FILTERS[name] for name in filter_names}

        print(f"[{year}] Mode A: daily edge-based ranking, metric={metric} ({len(filters_subset)} filters)...", flush=True)
        run_daily_ranking(
            filters=filters_subset,
            eth_dir=eth_dir,
            erc20_dir=erc20_dir,
            daily_dir=daily_dir,
            index_file=index_file,
            metrics=[metric],
            keep_cols=KEEP_COLS,
            start=start_date,
            end=end_date,
            top_n=TOP_N,
            dead_ads=DEAD_ADDRESSES,
        )

    print(f"[{year}] DONE in {time.time()-t0:.0f}s", flush=True)

print("\n\nAll years complete.", flush=True)
