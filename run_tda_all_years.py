"""Driver: rebuild persistence diagrams for the 6 ETH_only layers +
contract_txs_ERC20_only, across 3 filtration weights (tx_count, tx_value,
total_gas_fees where each is meaningful), training-window years only
(2020-2023).

Per this project's standing non-circular discipline (see STATUS.md's
Methodology Protocol): 2024-2025 stays untouched until a confirmatory
pipeline has run on this training-window result -- do not extend YEARS to
include the holdout without a deliberate, separate decision to do so.

TDA_CFG's edge_weight_col/ranking_metric apply globally per run_all()
call, so this issues 3 separate calls (one per weight), each with only
the layers where that weight is meaningful:
  - tx_count: all 7 layers (6 ETH_only + contract_txs_ERC20_only)
  - tx_value: simple_txs_ETH_only only (tx_value == 0 by definition for
    every contract_* filter, including contract_txs_ERC20_only)
  - total_gas_fees: the 6 ETH_only layers only (no gas info for ERC20
    events, so not contract_txs_ERC20_only)
This mirrors the exact same degenerate-combination exclusion already
applied in run_ranking_all_years.py.

Each weight group writes into its own run_results_V2_<bucket>_<YEAR>.json
(tx_count / tx_value / gasfees) rather than one merged file per year --
save_run_results() reads+rewrites the *entire* file on every save, so one
shared file would mean O(n^2) I/O across a year's 14 combos, and any
corrupted write would risk losing every already-completed combo in that
file, not just the interrupted one. Splitting by weight group bounds both
to at most 7 combos. See migrate_split_results.py for the one-off
migration of the original merged files. Deliberately NOT run_results_V1,
which every prior regime_detection Phase 1/2 finding was built from.

    python3 run_tda_all_years.py

Logs progress to stdout; redirect to a file if running unattended.
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, "functions")

from tad_ethereum_functions import run_all

YEARS = [2020, 2021, 2022, 2023]  # training window only -- see module docstring

BASE_LAYER_FILTERS = {
    "contract_txs_ETH_only": lambda d: (d["tx_value"] == 0) & (d["erc20"] == "ETH"),
    "simple_txs_ETH_only":   lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"),
    "contract_factory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["erc20"] == "ETH"),
    "contract_nonFactory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"),
    "contract_highInput_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"),
    "contract_mediumInput_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"),
    "contract_txs_ERC20_only": lambda d: (d["tx_value"] == 0) & (d["erc20"] != "ETH"),
}
ETH_ONLY_LAYER_NAMES = [n for n in BASE_LAYER_FILTERS if n != "contract_txs_ERC20_only"]

DAILY_TOP = 750

BASE_TDA_CFG = {
    "max_nodes":         1000,
    "homology_maxdim":   1,
    "distance_metric":   "wasserstein",
    "similarity_metric": "norm_similarity",
    "alpha":             9,
    "global_top":        0,
    "daily_top":         DAILY_TOP,
}

RESULTS_PREFIX_BASE = "run_results_V2"

# weight -> filename bucket, matching migrate_split_results.py's bucket_for()
WEIGHT_BUCKET = {
    "tx_count":       "tx_count",
    "tx_value":       "tx_value",
    "total_gas_fees": "gasfees",
}


def path_cfg_for(weight):
    return {
        "data_root":      Path("data"),
        "ranking_root":   Path("data/ranking"),
        "results_prefix": f"{RESULTS_PREFIX_BASE}_{WEIGHT_BUCKET[weight]}",
    }


def make_layers(names, weight_suffix):
    """{run_name: {layer_name: filter_fn}} for the given layer names, with
    run_name = f'{layer_name}{weight_suffix}_{DAILY_TOP}' (empty suffix for
    tx_count, matching the existing run_results_V1 naming convention)."""
    return {f"{name}{weight_suffix}_{DAILY_TOP}": {name: BASE_LAYER_FILTERS[name]} for name in names}


WEIGHT_GROUPS = [
    ("tx_count", list(BASE_LAYER_FILTERS), ""),
    ("tx_value", ["simple_txs_ETH_only"], "_value"),
    ("total_gas_fees", ETH_ONLY_LAYER_NAMES, "_gasfees"),
]

def existing_run_names(results_prefix, year):
    """run_name keys already present in <results_prefix>_<year>.json, so a
    rerun after an interruption skips (year, run_name) combos already done.
    run_all() itself has no such check -- it always recomputes and
    overwrites -- so this skip logic lives here in the driver instead."""
    results_file = Path(f"{results_prefix}_{year}.json")
    if not results_file.exists():
        return set()
    try:
        return set(json.loads(results_file.read_text()).keys())
    except (json.JSONDecodeError, OSError):
        return set()


for weight, layer_names, suffix in WEIGHT_GROUPS:
    print(f"{weight}: {len(layer_names)} layers -> {layer_names}")

for weight, layer_names, suffix in WEIGHT_GROUPS:
    t0 = time.time()
    print(f"\n{'='*80}\nWEIGHT GROUP: {weight} ({len(layer_names)} layers x {len(YEARS)} years)\n{'='*80}", flush=True)

    all_layers = make_layers(layer_names, suffix)
    tda_cfg = dict(BASE_TDA_CFG, edge_weight_col=weight, ranking_metric=weight)
    path_cfg = path_cfg_for(weight)

    for year in YEARS:
        done = existing_run_names(path_cfg["results_prefix"], year)
        missing = {name: filt for name, filt in all_layers.items() if name not in done}
        if not missing:
            print(f"  [{weight}] {year}: all {len(all_layers)} run(s) already present, skipping", flush=True)
            continue
        skipped = set(all_layers) - set(missing)
        if skipped:
            print(f"  [{weight}] {year}: skipping {sorted(skipped)} (already done), running {sorted(missing)}", flush=True)
        else:
            print(f"  [{weight}] {year}: running {sorted(missing)}", flush=True)

        run_all(
            years=[year],
            layers=missing,
            tda_cfg=tda_cfg,
            path_cfg=path_cfg,
        )

    print(f"[{weight}] DONE in {time.time()-t0:.0f}s", flush=True)

print("\n\nAll weight groups complete.", flush=True)
