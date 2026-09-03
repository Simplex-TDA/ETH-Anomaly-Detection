"""Driver: persistent homology over the trailing-7-day-ranked
ContractNonFactoryEthOnly edges (idea 13's DuckDB rewrite,
run_trailing_rank_duckdb.py -- ranking/edge-selection only, no topology).

This is the missing step flagged in STATUS.md's "Idea 13" entry: node
selection is already done (top-750 trailing-summed-tx_count edge pairs
per day define the qualifying node set; the actual per-day graph is that
day's own tx_count edges restricted to those nodes -- see
run_trailing_rank_duckdb.py's process_chunk). This script feeds that
already-selected edge data into the same graph-building + geodesic-
densification + Ripser pipeline every other layer in this project uses,
so the result is directly comparable to `ContractNonFactoryEthOnly750`
from the daily_top sweep (idea 3) -- same max_nodes (2500, matching that
sweep's rebuild, not the older max_nodes=1000 runs), same
similarity_metric/alpha, same maxdim. The only axis that differs is node
selection: trailing-7-day-summed ranking here vs. single-day ranking
there.

    python3 scripts/runs/run_tda_trailing_window.py

Logs progress to stdout; redirect to a file if running unattended, e.g.:
    nohup python3 scripts/runs/run_tda_trailing_window.py > logs/run_tda_trailing_window.log 2>&1 &
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/
os.chdir(ROOT)
sys.path.insert(0, "functions")

import numpy as np
import pandas as pd
import networkx as nx
import tad_ethereum_functions as tef
from tad_ethereum_functions import (
    run_tda_pipeline, run_distance_series, extract_daily_features,
    add_temporal_features, compute_indices, save_run_results,
)


def _build_graph_fast(day_df, edge_weight_col='tx_count'):
    """Same vectorized replacement used in run_tda_daily_top_sweep.py --
    see that script's docstring for the full rationale/validation. Applies
    equally here: this data has the same real-world edge-count spikes."""
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

# ── same per-day hard timeout as run_tda_daily_top_sweep.py -- see that
# script's comment block for the full incident history (MEMORY.md's
# "Ripser pathological days" note). Reused verbatim since this data can
# hit the same real-activity spikes (2024-2025 especially). ──
import multiprocessing as mp

DAY_TIMEOUT_SECONDS = 300
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
MAX_NODES = 2500
RUN_NAME = "contract_nonFactory_ETH_only_trailing7d_750"
EDGES_DIR = ROOT / "results" / "exploratory" / "trailing_rank_by_year"
RESULTS_PREFIX = "results/run_results_V2_trailing_window_tx_count"

TDA_CFG = {
    "edge_weight_col":   "tx_count",
    "max_nodes":         MAX_NODES,
    "homology_maxdim":   1,
    "distance_metric":   "wasserstein",
    "similarity_metric": "norm_similarity",
    "alpha":             9,
    "ranking_metric":    "tx_count",
    "global_top":        0,
    "daily_top":         750,  # trailing-window top-N-edges cutoff (see run_trailing_rank_duckdb.py)
}

t_start = time.time()

for year in YEARS:
    results_file = Path(f"{RESULTS_PREFIX}_{year}.json")
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if RUN_NAME in existing:
            print(f"[{time.time()-t_start:6.0f}s] {year}: already done, skipping", flush=True)
            continue

    edges_file = EDGES_DIR / f"edges_{year}.parquet"
    if not edges_file.exists():
        print(f"[{time.time()-t_start:6.0f}s] {year}: {edges_file} not found, skipping", flush=True)
        continue

    filtered_df = pd.read_parquet(edges_file)
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    dates = sorted(filtered_df["date"].unique())
    print(f"[{time.time()-t_start:6.0f}s] {year}: loaded {len(filtered_df):,} rows, {len(dates)} days", flush=True)

    daily_edge_nums = (
        filtered_df[['date', 'from_addr']]
        .groupby('date').count()
        .rename(columns={'from_addr': 'edges'})
    )
    daily_node_nums = (
        filtered_df[['date', 'from_addr', 'to_addr']]
        .groupby('date').agg(
            nodes=('from_addr',
                   lambda s: pd.concat(
                       [s, filtered_df.loc[s.index, 'to_addr']]
                   ).nunique())
        )
    )

    t0 = time.time()
    pd_series, features_series, stats_df = run_tda_pipeline(
        df=filtered_df,
        edge_weight_col=TDA_CFG["edge_weight_col"],
        layer_filters=None,
        max_nodes=TDA_CFG["max_nodes"],
        maxdim=TDA_CFG["homology_maxdim"],
        similarity_metric=TDA_CFG["similarity_metric"],
        alpha=TDA_CFG["alpha"],
    )
    tda_time = (time.time() - t0) / 60

    t0 = time.time()
    distance_series = run_distance_series(pd_series, distance_metric=TDA_CFG["distance_metric"])
    distance_aggregation_time = (time.time() - t0) / 60

    tda_index_features = compute_indices(add_temporal_features(extract_daily_features(pd_series)))

    config = dict(TDA_CFG, layers=["contract_nonFactory_ETH_only"], year=year, ranking_dir="duckdb_trailing7d")

    run = {
        'config':                    config,
        'num_edges':                 len(filtered_df),
        'num_nodes':                 pd.concat([filtered_df['from_addr'], filtered_df['to_addr']]).nunique(),
        'daily_edge_nums':           daily_edge_nums,
        'daily_node_nums':           daily_node_nums,
        'pd_series':                 pd_series,
        'distance_series':           distance_series,
        'tda_index_features':        tda_index_features,
        'stats_df':                  stats_df,
        'features_series':           features_series,
        'layer_names':               ["contract_nonFactory_ETH_only"],
        'filtering_time':            0.0,
        'tda_time':                  tda_time,
        'distance_aggregation_time': distance_aggregation_time,
    }

    save_run_results(results_file, RUN_NAME, run)
    print(f"[{time.time()-t_start:6.0f}s] {year}: saved {len(pd_series)}/{len(dates)} days -> {results_file}", flush=True)

print(f"\n\nAll years complete in {time.time()-t_start:.0f}s.", flush=True)
print("DONE.", flush=True)
