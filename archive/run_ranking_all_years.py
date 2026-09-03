"""Driver: daily ranking (Mode A, edge-based) + efficient daily centrality
(Mode B) for 2020-2025. No global ranking (per the user's explicit scope).

Resumable by construction -- run_daily_ranking and
run_daily_centrality_ranking_efficient both track completed
(filter, ranking_metric, date) triples in the daily parquet files
themselves and skip anything already there. Safe to re-run if interrupted.

    python3 run_ranking_all_years.py

Logs progress to stdout; redirect to a file if running unattended.

IMPORTANT, found via a small-scale test before queuing the full run: some
(filter, weight_col) combinations are degenerate by construction --
tx_value is identically 0 for every contract_* filter (that's the filter's
own definition), and total_gas_fees is identically 0 for the *_ERC20_only
filters (ERC20 transfer events don't carry gas info in Xatu's schema, see
2_ranking.ipynb's config cell). Ranking/weighting by a metric that's
uniformly zero across an entire filter is meaningless regardless of which
centrality algorithm is used -- nx.clustering's weighted formula divides
by zero on these graphs specifically (confirmed empirically: every
failure traced back to exactly these combinations, nothing else), and
even where a metric doesn't outright crash (page_rank/strength/k_core
tolerate an all-zero-weight graph), the result would still be an
uninformative, arbitrary tie-break rather than a real ranking. These
combinations are excluded below rather than computed and silently
producing degenerate output.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, "functions")

import pandas as pd
from ranking_functions import (
    check_directories,
    run_daily_ranking,
    run_daily_centrality_ranking_efficient,
)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TOP_N = 2000
DEAD_ADDRESSES = [None, "\\N"]

CENTRALITY_METRICS = ["page_rank", "strength", "k_core", "clustering"]

# Same 14 filters as 2_ranking.ipynb (6 _ALL, 6 _ETH_only, 2 new _ERC20_only).
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

CONTRACT_FILTER_NAMES = [f for f in ALL_FILTERS if f.startswith("contract_")]  # tx_value == 0 by definition
ERC20_ONLY_FILTER_NAMES = [f for f in ALL_FILTERS if f.endswith("_ERC20_only")]  # no gas info

# One (metric, applicable-filter-subset) pair per valid combination.
METRIC_FILTER_GROUPS = [
    ("tx_count", list(ALL_FILTERS)),  # always meaningful, every filter
    ("tx_value", [f for f in ALL_FILTERS if f not in CONTRACT_FILTER_NAMES]),  # simple_txs_* only
    ("total_gas_fees", [f for f in ALL_FILTERS if f not in ERC20_ONLY_FILTER_NAMES]),  # everything except *_ERC20_only
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

        print(f"[{year}] Mode B: efficient daily centrality ranking, weight={metric} "
              f"({CENTRALITY_METRICS}, {len(filters_subset)} filters)...", flush=True)
        run_daily_centrality_ranking_efficient(
            filters=filters_subset,
            eth_dir=eth_dir,
            erc20_dir=erc20_dir,
            daily_dir=daily_dir,
            index_file=index_file,
            centrality_metrics=CENTRALITY_METRICS,
            weight_cols=[metric],
            keep_cols=KEEP_COLS,
            start=start_date,
            end=end_date,
            top_n=TOP_N,
            dead_ads=DEAD_ADDRESSES,
        )

    print(f"[{year}] DONE in {time.time()-t0:.0f}s", flush=True)

print("\n\nAll years complete.", flush=True)
