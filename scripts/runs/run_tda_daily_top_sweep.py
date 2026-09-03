"""Driver: daily_top (node-count) sensitivity sweep.

Every layer/arm in this project has so far been built on a single fixed
daily_top=750 (data/regime_detection's HANDOFF.md "Next step" section) --
never varied. This rebuilds persistence diagrams for 5 layers x 4 daily_top
values (500, 750, 1000, 1250), so downstream feature/AUC comparisons can
measure how much this parameter actually matters.

max_nodes raised from 1000 to 2500 for this whole sweep. Reason: checking
the raw (pre-densification-cap) daily node counts already recorded in the
existing daily_top=750 / max_nodes=1000 runs showed the old cap already
truncated the busiest ~1% of days for 2 of these 5 layers (contract_
nonFactory_ETH_only hit 1252 raw nodes in 2024; simple_txs_ETH_only hit
1162 in 2025) -- so max_nodes=1000 was not actually a safe no-op even at
750, and would truncate far more at 1250. 2500 gives headroom above the
worst case extrapolated for daily_top=1250 at the same truncation ratio
observed at 750 (~1.7x).

daily_top=750 is rebuilt fresh here (not reused from the existing
run_results_V2_*.json) so every point in the sweep shares the same
max_nodes=2500 and is a clean, directly-comparable baseline -- the existing
_750 diagrams used max_nodes=1000 and are not comparable 1:1 for this
purpose.

Layers (5, matching the strongest/most-relevant holdout results from the
Arm-I full-battery testing -- see STATUS.md "The 18 under-tested layers"
and "V2 weight-type layers" sections):
  - simple_txs_ETH_only            (tx_count)
  - simple_txs_ETH_only            (tx_value)  -- the one tx_value-weighted
                                                   layer with a confirmed
                                                   result; doubles as a
                                                   weight-type check
  - contract_nonFactory_ETH_only   (tx_count)  -- strongest single layer in
                                                   the whole 18-layer battery
  - contract_mediumInput_ETH_only  (tx_count)
  - contract_highInput_ETH_only    (tx_count)

Written to its own results files (run_results_V2_dailytop_sweep_<bucket>_
<year>.json), never touching the existing run_results_V2_<bucket>_<year>.json
files the rest of the project's confirmed findings are built from.

    python3 scripts/runs/run_tda_daily_top_sweep.py

Logs progress to stdout; redirect to a file if running unattended, e.g.:
    nohup python3 scripts/runs/run_tda_daily_top_sweep.py > logs/run_tda_daily_top_sweep.log 2>&1 &
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/runs/)
os.chdir(ROOT)
sys.path.insert(0, "functions")

import numpy as np
import pandas as pd
import networkx as nx
import tad_ethereum_functions as tef
from tad_ethereum_functions import run_all


def _build_graph_fast(day_df, edge_weight_col='tx_count'):
    """Drop-in, behavior-identical replacement for tad_ethereum_functions.
    build_graph, vectorized instead of row-by-row .iterrows(). The original
    is O(rows) but with a very large Python-level constant factor (per-row
    G.has_edge()/G.add_edge() calls) -- fine for the ~1-15k edges/day this
    project has mostly seen, but the first daily_top=1250 attempt (2026-08-14
    launch) stalled for 8+ hours and was killed on Aug 2025 dates, where the
    OR-inclusion edge-survival rule in load_filtered_graph plus a genuine
    real-world activity spike pushed a single day's edge count into a range
    where the row-by-row loop became the dominant cost. Verified equivalent
    to the original across randomized trials (zero/negative/NaN weights,
    both u/v orderings, self-loops) with a 105x speedup on a 200k-row
    synthetic benchmark. Monkeypatched into tad_ethereum_functions below
    rather than editing that shared, already-validated module in place.
    """
    u = day_df['from_addr'].to_numpy(dtype=object)
    v = day_df['to_addr'].to_numpy(dtype=object)
    w = day_df[edge_weight_col].to_numpy(dtype=float)
    bad = (w <= 0) | np.isnan(w)
    w = np.where(bad, 1e-6, w)
    a = np.where(u <= v, u, v)
    b = np.where(u <= v, v, u)
    tmp = pd.DataFrame({'a': a, 'b': b, 'w': w})
    summed = tmp.groupby(['a', 'b'], sort=False, as_index=False)['w'].sum()
    G = nx.Graph()
    G.add_weighted_edges_from(zip(summed['a'], summed['b'], summed['w']))
    return G


tef.build_graph = _build_graph_fast


# ─────────────────────────────────────────────────────────────────────────
# Per-day hard timeout, added after the 2026-08-18 relaunch also stalled --
# on the *same* Aug-2025 day as the original (pre-build_graph-fix) attempt.
# Measured directly: that day's OR-inclusion edge-survival rule (in
# load_filtered_graph) realizes ~51,000 raw nodes / ~76,000 edges before
# max_nodes=2500 trims it down -- build_graph_fast handles that fine, but
# geodesic_densification's Dijkstra + Ripser's persistence-homology
# computation on the resulting (still dense, real-world) 2500-node subgraph
# apparently doesn't: RSS climbed to 7.7GB and the whole machine's swap
# went from unremarkable to 12.85/14.3GB used (thrashing) while it sat on
# that one day for 1h40m+ with zero progress, mirroring the original run's
# 8+ hour silent stall before it was killed. Per-day wall-clock cap here,
# enforced via a forked subprocess (Ripser is a C extension -- a Python-level
# signal/exception can't reliably interrupt it mid-call, so the only robust
# way to bound its runtime is to run it somewhere killable from outside).
# Timed-out days are skipped and logged (day/run/node-count), matching the
# existing "not enough data" skip path elsewhere in this pipeline -- an
# honest, audited exclusion of a handful of pathological days, not a silent
# one, out of otherwise-365-day years.
# ─────────────────────────────────────────────────────────────────────────
import multiprocessing as mp

DAY_TIMEOUT_SECONDS = 300  # 5 min hard cap on any single day's TDA computation
_orig_compute_day_pd = tef.compute_day_pd


def _compute_day_pd_worker(day_df, edge_weight_col, layer_filters, max_nodes,
                            maxdim, dist_func, alpha, result_queue):
    try:
        result_queue.put(('ok', _orig_compute_day_pd(
            day_df, edge_weight_col=edge_weight_col, layer_filters=layer_filters,
            max_nodes=max_nodes, maxdim=maxdim, dist_func=dist_func, alpha=alpha,
        )))
    except Exception as e:
        result_queue.put(('error', repr(e)))


def _compute_day_pd_with_timeout(day_df, edge_weight_col, layer_filters, max_nodes,
                                  maxdim, dist_func='norm_similarity', alpha=9):
    date_str = str(day_df['date'].iloc[0].date()) if len(day_df) else 'unknown-date'
    n_addrs = pd.concat([day_df['from_addr'], day_df['to_addr']]).nunique() if len(day_df) else 0

    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(
        target=_compute_day_pd_worker,
        args=(day_df, edge_weight_col, layer_filters, max_nodes, maxdim, dist_func, alpha, q),
    )
    p.start()
    p.join(DAY_TIMEOUT_SECONDS)

    if p.is_alive():
        print(f"  [SKIP-TIMEOUT] {date_str}: exceeded {DAY_TIMEOUT_SECONDS}s "
              f"({len(day_df)} rows, {n_addrs} raw addrs) -- killing and skipping this day", flush=True)
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        q.close()
        return None, {'skipped_timeout': True, 'raw_addrs': int(n_addrs), 'raw_rows': len(day_df)}, None

    if not q.empty():
        status, payload = q.get()
        p.join()
        q.close()
        if status == 'ok':
            return payload
        print(f"  [SKIP-ERROR] {date_str}: worker raised {payload}", flush=True)
        return None, {'skipped_error': payload}, None

    p.join()
    q.close()
    print(f"  [SKIP-DIED] {date_str}: worker process died with no result "
          f"(exitcode={p.exitcode}, {len(day_df)} rows, {n_addrs} raw addrs) -- likely OOM, skipping this day", flush=True)
    return None, {'skipped_died': True, 'exitcode': p.exitcode}, None


tef.compute_day_pd = _compute_day_pd_with_timeout

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAILY_TOP_VALUES = [500, 750, 1000, 1250]
MAX_NODES = 2500

LAYER_FILTERS = {
    "simple_txs_ETH_only":           lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"),
    "contract_nonFactory_ETH_only":  lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"),
    "contract_mediumInput_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"),
    "contract_highInput_ETH_only":   lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"),
}

BASE_TDA_CFG = {
    "max_nodes":         MAX_NODES,
    "homology_maxdim":   1,
    "distance_metric":   "wasserstein",
    "similarity_metric": "norm_similarity",
    "alpha":             9,
    "global_top":        0,
}

RESULTS_PREFIX_BASE = "results/run_results_V2_dailytop_sweep"

# weight -> (bucket suffix for filename, layer names that get this weight,
# run_name suffix). simple_txs_ETH_only gets both weights (tx_count is the
# default/main comparison, tx_value is the deliberate weight-type check);
# the other 4 are tx_count only, matching the project's existing finding
# that tx_count generalizes best on holdout among the weight types tested.
WEIGHT_GROUPS = [
    ("tx_count", "tx_count", list(LAYER_FILTERS), ""),
    ("tx_value", "tx_value", ["simple_txs_ETH_only"], "_value"),
]


def path_cfg_for(bucket):
    return {
        "data_root":      Path("data"),
        "ranking_root":   Path("data/ranking"),
        "results_prefix": f"{RESULTS_PREFIX_BASE}_{bucket}",
    }


def make_layers(names, weight_suffix, daily_top):
    return {f"{name}{weight_suffix}_{daily_top}": {name: LAYER_FILTERS[name]} for name in names}


def existing_run_names(results_prefix, year):
    """Skip (year, run_name) combos already present, so an interrupted run
    can be resumed by just re-launching this script."""
    results_file = Path(f"{results_prefix}_{year}.json")
    if not results_file.exists():
        return set()
    try:
        return set(json.loads(results_file.read_text()).keys())
    except (json.JSONDecodeError, OSError):
        return set()


for weight, bucket, layer_names, suffix in WEIGHT_GROUPS:
    print(f"{weight}: {len(layer_names)} layers x {len(DAILY_TOP_VALUES)} daily_top values -> {layer_names}")
total_combos = sum(len(layer_names) for _, _, layer_names, _ in WEIGHT_GROUPS) * len(DAILY_TOP_VALUES) * len(YEARS)
print(f"Total (weight, layer, daily_top, year) combos: {total_combos}\n")

t_start = time.time()

for weight, bucket, layer_names, suffix in WEIGHT_GROUPS:
    for daily_top in DAILY_TOP_VALUES:
        t0 = time.time()
        print(f"\n{'='*80}\nWEIGHT={weight} DAILY_TOP={daily_top} MAX_NODES={MAX_NODES} "
              f"({len(layer_names)} layers x {len(YEARS)} years)\n{'='*80}", flush=True)

        all_layers = make_layers(layer_names, suffix, daily_top)
        tda_cfg = dict(BASE_TDA_CFG, edge_weight_col=weight, ranking_metric=weight, daily_top=daily_top)
        path_cfg = path_cfg_for(bucket)

        for year in YEARS:
            done = existing_run_names(path_cfg["results_prefix"], year)
            missing = {name: filt for name, filt in all_layers.items() if name not in done}
            if not missing:
                print(f"  [{weight}/{daily_top}] {year}: all {len(all_layers)} run(s) already present, skipping", flush=True)
                continue
            skipped = set(all_layers) - set(missing)
            if skipped:
                print(f"  [{weight}/{daily_top}] {year}: skipping {sorted(skipped)} (already done), running {sorted(missing)}", flush=True)
            else:
                print(f"  [{weight}/{daily_top}] {year}: running {sorted(missing)}", flush=True)

            run_all(
                years=[year],
                layers=missing,
                tda_cfg=tda_cfg,
                path_cfg=path_cfg,
            )

        print(f"[{weight}/{daily_top}] DONE in {time.time()-t0:.0f}s", flush=True)

print(f"\n\nAll weight/daily_top groups complete in {time.time()-t_start:.0f}s.", flush=True)
