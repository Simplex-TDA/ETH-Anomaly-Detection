"""Driver: rebuild EVERY persistence diagram this project relies on, from
the FINAL-corrected (deduplicated) raw data and ranking output --
the 8 original V1 layers (this project's central, most-vetted findings --
Arms C/D/E/G/I/U and everything built on top of them) AND the 14 V2-rebuild
ETH-only/weighted layers, across all 6 years (2020-2025).

Root cause this rebuild fixes: canonical_execution_transaction/_block/
_erc20_transfers are all ReplicatedReplacingMergeTree tables -- a SELECT
without FINAL can see un-merged duplicate rows depending on background
merge timing. No query anywhere in this project's history (fetcher,
ranking, or otherwise) ever used FINAL until this session. Confirmed
empirically: the same block range gave 196,499 (with FINAL, correct)
vs. 327,939 (without) vs. 427,342 (the original V2 fetch) raw transactions.
This affects every TDA-derived finding in the project; it does NOT affect
financial-only results (Arm A, Arm J, GARCH), which never touch Xatu.

V1's 8 layers, exact filter definitions and TDA_CFG copied verbatim from
baseline_topology_engine/3_tda.ipynb (the original, byte-verified-identical
reference copy) -- single-layer mode, tx_count weight only, matching what
every V1-based confirmatory result in this project was actually built from.
Written to run_results_V1_<YEAR>.json (replacing the archived, pre-FINAL
versions in archive_pre_final_fix/).

V2's 14 (layer, weight) combos, unchanged from run_tda_all_years.py.
Written to run_results_V2_<bucket>_<YEAR>.json.

Resumability: existing_run_names() skips any (year, run_name) combo already
present in this fresh run's own output files -- since those files were
moved to archive_pre_final_fix/ rather than left in place, this rebuild
starts genuinely from scratch, not confused by old pre-FINAL data.

    python3 run_tda_all_layers_FINAL.py

Logs progress to stdout; redirect to a file if running unattended.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/runs/)
os.chdir(ROOT)
sys.path.insert(0, "functions")

from tad_ethereum_functions import run_all

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAILY_TOP = 750

# ═══════════════════════════════════════════════════════════════════════════
# V1: the 8 original layers -- exact copy of baseline_topology_engine/3_tda.ipynb
# ═══════════════════════════════════════════════════════════════════════════

V1_LAYERS = {
    'contract_txs_ETH_only_750':     {'contract_txs_ETH_only':     lambda d: (d['tx_value'] == 0) & (d['erc20'] == 'ETH')},
    'simple_txs_ETH_only_750':       {'simple_txs_ETH_only':       lambda d: (d['tx_value'] >  0) & (d['erc20'] == 'ETH')},
    'contract_txs_ALL_750':          {'contract_txs_ALL':          lambda d: (d['tx_value'] == 0)},
    'simple_txs_ALL_750':            {'simple_txs_ALL':            lambda d: (d['tx_value'] >  0)},
    'contract_factory_ALL_750':      {'contract_factory_ALL':      lambda d: (d['tx_value'] == 0) & (d['total_input_bytes'] >= 100)},
    'contract_nonFactory_ALL_750':   {'contract_nonFactory_ALL':   lambda d: (d['tx_value'] == 0) & (d['total_input_bytes'] < 100)},
    'contract_highInput_ALL_750':    {'contract_highInput_ALL':    lambda d: (d['tx_value'] == 0) & (d['total_input_bytes'] >= 500)},
    'contract_mediumInput_ALL_750':  {'contract_mediumInput_ALL':  lambda d: (d['tx_value'] == 0) & (d['total_input_bytes'] >= 100) & (d['total_input_bytes'] < 500)},
}

V1_TDA_CFG = {
    'edge_weight_col':   'tx_count',
    'max_nodes':         1000,
    'homology_maxdim':   1,
    'distance_metric':   'wasserstein',
    'similarity_metric': 'norm_similarity',
    'alpha':             9,
    'ranking_metric':    'tx_count',
    'global_top':        0,
    'daily_top':         DAILY_TOP,
}

V1_PATH_CFG = {
    'data_root':      Path('data'),
    'ranking_root':   Path('data/ranking'),
    'results_prefix': 'results/run_results_V1',
}

# ═══════════════════════════════════════════════════════════════════════════
# V2: the 14 (layer, weight) combos -- unchanged from run_tda_all_years.py
# ═══════════════════════════════════════════════════════════════════════════

V2_BASE_LAYER_FILTERS = {
    "contract_txs_ETH_only":        lambda d: (d["tx_value"] == 0) & (d["erc20"] == "ETH"),
    "simple_txs_ETH_only":          lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"),
    "contract_factory_ETH_only":    lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["erc20"] == "ETH"),
    "contract_nonFactory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"),
    "contract_highInput_ETH_only":  lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"),
    "contract_mediumInput_ETH_only":lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"),
    "contract_txs_ERC20_only":      lambda d: (d["tx_value"] == 0) & (d["erc20"] != "ETH"),
}
ETH_ONLY_LAYER_NAMES = [n for n in V2_BASE_LAYER_FILTERS if n != "contract_txs_ERC20_only"]

V2_BASE_TDA_CFG = {
    "max_nodes":         1000,
    "homology_maxdim":   1,
    "distance_metric":   "wasserstein",
    "similarity_metric": "norm_similarity",
    "alpha":             9,
    "global_top":        0,
    "daily_top":         DAILY_TOP,
}

V2_RESULTS_PREFIX_BASE = "results/run_results_V2"
V2_WEIGHT_BUCKET = {"tx_count": "tx_count", "tx_value": "tx_value", "total_gas_fees": "gasfees"}

V2_WEIGHT_GROUPS = [
    ("tx_count",        list(V2_BASE_LAYER_FILTERS), ""),
    ("tx_value",        ["simple_txs_ETH_only"],      "_value"),
    ("total_gas_fees",  ETH_ONLY_LAYER_NAMES,          "_gasfees"),
]


def v2_path_cfg_for(weight):
    return {
        "data_root":      Path("data"),
        "ranking_root":   Path("data/ranking"),
        "results_prefix": f"{V2_RESULTS_PREFIX_BASE}_{V2_WEIGHT_BUCKET[weight]}",
    }


def v2_make_layers(names, weight_suffix):
    return {f"{name}{weight_suffix}_{DAILY_TOP}": {name: V2_BASE_LAYER_FILTERS[name]} for name in names}


# ═══════════════════════════════════════════════════════════════════════════
# Shared resumability helper
# ═══════════════════════════════════════════════════════════════════════════

def existing_run_names(results_prefix, year):
    results_file = Path(f"{results_prefix}_{year}.json")
    if not results_file.exists():
        return set()
    try:
        return set(json.loads(results_file.read_text()).keys())
    except (json.JSONDecodeError, OSError):
        return set()


# ═══════════════════════════════════════════════════════════════════════════
# Run V1 (8 layers, tx_count only)
# ═══════════════════════════════════════════════════════════════════════════

t_start = time.time()
print(f"\n{'='*80}\nV1: 8 original layers, tx_count weight, {len(YEARS)} years\n{'='*80}", flush=True)

for year in YEARS:
    done = existing_run_names(V1_PATH_CFG["results_prefix"], year)
    missing = {name: filt for name, filt in V1_LAYERS.items() if name not in done}
    if not missing:
        print(f"  [V1] {year}: all {len(V1_LAYERS)} run(s) already present, skipping", flush=True)
        continue
    skipped = set(V1_LAYERS) - set(missing)
    if skipped:
        print(f"  [V1] {year}: skipping {sorted(skipped)} (already done), running {sorted(missing)}", flush=True)
    else:
        print(f"  [V1] {year}: running {sorted(missing)}", flush=True)

    run_all(
        years=[year],
        layers=missing,
        tda_cfg=V1_TDA_CFG,
        path_cfg=V1_PATH_CFG,
    )

print(f"[V1] DONE in {time.time()-t_start:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# Run V2 (14 layer x weight combos)
# ═══════════════════════════════════════════════════════════════════════════

for weight, layer_names, suffix in V2_WEIGHT_GROUPS:
    print(f"{weight}: {len(layer_names)} layers -> {layer_names}")

for weight, layer_names, suffix in V2_WEIGHT_GROUPS:
    t0 = time.time()
    print(f"\n{'='*80}\nV2 WEIGHT GROUP: {weight} ({len(layer_names)} layers x {len(YEARS)} years)\n{'='*80}", flush=True)

    all_layers = v2_make_layers(layer_names, suffix)
    tda_cfg = dict(V2_BASE_TDA_CFG, edge_weight_col=weight, ranking_metric=weight)
    path_cfg = v2_path_cfg_for(weight)

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

print(f"\n\nAll layers (V1 + V2) complete in {time.time()-t_start:.0f}s.", flush=True)
