"""Driver: rank-band decomposition pilot for 2 layers (contract_nonFactory_
ETH_only, simple_txs_ETH_only tx_count) -- see graph_prep.md and the
daily_top sweep's Arm-I-equiv results for the background. Instead of a
single top-N cutoff, each layer is split into 3 disjoint rank bands, each
built as its own separate graph/persistence-diagram per day. Bands chosen
from the edge-rank decay diagnostic (band_diagnostic_v2.py), pinned to
where each layer's log-log slope actually bends, not round numbers:

  contract_nonFactory_ETH_only  (saturates ~rank 1250-1500):
    band1 = ranks (0, 250]     -- steep hub tier
    band2 = ranks (250, 1250]  -- real but weakening signal, ends ~where
                                  the layer's own saturation point is
    band3 = ranks (1250, 2000] -- past-saturation zone; this band is the
                                  direct empirical test of the saturation
                                  finding -- expected to add ~nothing

  simple_txs_ETH_only tx_count  (no saturation observed to rank 20,000+):
    band1 = ranks (0, 500]
    band2 = ranks (500, 1500]
    band3 = ranks (1500, 2000] -- genuinely untested territory (this
                                  project has never gone past daily_top=
                                  1250); this band tests whether the
                                  no-saturation reading translates into
                                  real added predictive signal

Capped at rank 2000 for this pilot (not deeper) because the existing daily
ranking files were only computed to TOP_N=2000 (run_ranking_all_years.py);
staying within that avoids a new ranking-computation pass. If band3 shows
real signal for simple_txs, extending past 2000 is the natural follow-up.

Reuses the vectorized build_graph and per-day timeout+GUDHI-safe patterns
from run_tda_daily_top_sweep.py verbatim (band graphs are a strict subset
of that pipeline's node-selection step -- same downstream risk profile).

    python3 run_tda_band_sweep.py
    nohup python3 scripts/runs/run_tda_band_sweep.py > logs/run_tda_band_sweep.log 2>&1 &
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
from tad_ethereum_functions import run_all, load_filtered_graph as _orig_load_filtered_graph
from tqdm import tqdm


# ── same vectorized build_graph replacement as the daily_top sweep ────────
def _build_graph_fast(day_df, edge_weight_col='tx_count'):
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


# ── load_filtered_graph, extended to accept a (lo, hi] rank band as
# daily_top_num instead of only a single ceiling. Everything else (global
# top-node handling, edge AND-survival logic, streaming/filtering) is
# identical to the trusted original -- only the daily rank mask changes. ──
def _load_filtered_graph_banded(
    eth_dir, erc20_dir, global_file, daily_dir, filters, ranking_metric,
    global_top_num, daily_top_num, start_date=None, end_date=None,
):
    if not isinstance(daily_top_num, tuple):
        return _orig_load_filtered_graph(
            eth_dir, erc20_dir, global_file, daily_dir, filters, ranking_metric,
            global_top_num, daily_top_num, start_date, end_date,
        )
    band_lo, band_hi = daily_top_num

    tqdm.write('[1/4] Loading global top nodes ...')
    global_nodes = set()
    if global_file.exists():
        gdf = pd.read_parquet(global_file)
        mask = (gdf['filter'].isin(filters) & (gdf['ranking_metric'] == ranking_metric)
                & (gdf['bottom_rank'] <= global_top_num))
        global_nodes = set(gdf.loc[mask, 'address'].dropna().unique())
    tqdm.write(f'    {len(global_nodes):,} global top nodes')

    tqdm.write(f'[2/4] Loading daily top nodes (band ({band_lo}, {band_hi}]) ...')
    daily_nodes = {}
    for fpath in sorted(daily_dir.glob('*.parquet')):
        ddf = pd.read_parquet(fpath)
        ddf['date'] = pd.to_datetime(ddf['date'])
        if start_date is not None:
            ddf = ddf[ddf['date'] >= start_date]
        if end_date is not None:
            ddf = ddf[ddf['date'] <= end_date]
        if ddf.empty:
            continue
        mask = (ddf['filter'].isin(filters) & (ddf['ranking_metric'] == ranking_metric)
                & (ddf['bottom_rank'] > band_lo) & (ddf['bottom_rank'] <= band_hi))
        relevant = ddf.loc[mask, ['date', 'address']].dropna()
        for dt, grp in relevant.groupby('date', sort=False):
            dt = pd.Timestamp(dt)
            daily_nodes.setdefault(dt, set()).update(grp['address'].unique())
    tqdm.write(f'    {len(daily_nodes):,} dates with daily top nodes')

    tqdm.write('[3/4] Streaming source files ...')
    result_parts = []
    for folder, token_label in [(eth_dir, 'ETH'), (erc20_dir, None)]:
        for fpath in sorted(folder.glob('*.parquet')):
            chunk = pd.read_parquet(fpath)
            chunk['date'] = pd.to_datetime(chunk['date'])
            if token_label is not None:
                chunk['erc20'] = token_label
            if start_date is not None:
                chunk = chunk[chunk['date'] >= start_date]
            if end_date is not None:
                chunk = chunk[chunk['date'] <= end_date]
            if chunk.empty:
                continue
            frm = chunk['from_addr'].to_numpy(dtype=object)
            to = chunk['to_addr'].to_numpy(dtype=object)
            dt = chunk['date'].to_numpy()
            frm_global = pd.Series(frm).isin(global_nodes).to_numpy()
            to_global = pd.Series(to).isin(global_nodes).to_numpy()
            frm_daily = np.zeros(len(frm), dtype=bool)
            to_daily = np.zeros(len(to), dtype=bool)
            for unique_dt in chunk['date'].unique():
                day_set = daily_nodes.get(pd.Timestamp(unique_dt), set())
                if not day_set:
                    continue
                row_mask = (dt == unique_dt)
                frm_daily[row_mask] = pd.Series(frm[row_mask]).isin(day_set).to_numpy()
                to_daily[row_mask] = pd.Series(to[row_mask]).isin(day_set).to_numpy()
            frm_ok = frm_global | frm_daily
            to_ok = to_global | to_daily
            keep = frm_ok & to_ok
            if keep.any():
                result_parts.append(chunk.loc[keep].copy())

    tqdm.write('[4/4] Combining ...')
    if not result_parts:
        tqdm.write('    [WARN] No rows survived the filter.')
        return pd.DataFrame(columns=['date', 'from_addr', 'to_addr', 'erc20',
                                      'tx_count', 'tx_value', 'max_value',
                                      'total_gas_price', 'total_input_bytes',
                                      'min_input_bytes'])
    result = pd.concat(result_parts, ignore_index=True)
    tqdm.write(f'    Total rows: {len(result):,}')
    return result


tef.load_filtered_graph = _load_filtered_graph_banded


# ── same per-day fork-subprocess timeout wrapper as the daily_top sweep ───
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
              f"({len(day_df)} rows, {n_addrs} raw addrs)", flush=True)
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
    print(f"  [SKIP-DIED] {date_str}: worker died (exitcode={p.exitcode})", flush=True)
    return None, {'skipped_died': True, 'exitcode': p.exitcode}, None


tef.compute_day_pd = _compute_day_pd_with_timeout

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MAX_NODES = 2500

LAYER_FILTERS = {
    "simple_txs_ETH_only":          lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"),
    "contract_nonFactory_ETH_only": lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"),
}

BANDS = {
    "contract_nonFactory_ETH_only": {"band1": (0, 250), "band2": (250, 1250), "band3": (1250, 2000)},
    "simple_txs_ETH_only":          {"band1": (0, 500), "band2": (500, 1500), "band3": (1500, 2000)},
}

BASE_TDA_CFG = {
    "max_nodes":         MAX_NODES,
    "homology_maxdim":   1,
    "distance_metric":   "wasserstein",
    "similarity_metric": "norm_similarity",
    "alpha":             9,
    "global_top":        0,
    "ranking_metric":    "tx_count",
    "edge_weight_col":   "tx_count",
}

RESULTS_PREFIX = "results/run_results_V2_bandsweep_tx_count"


def path_cfg_for():
    return {
        "data_root":      Path("data"),
        "ranking_root":   Path("data/ranking"),
        "results_prefix": RESULTS_PREFIX,
    }


def existing_run_names(results_prefix, year):
    results_file = Path(f"{results_prefix}_{year}.json")
    if not results_file.exists():
        return set()
    try:
        return set(json.loads(results_file.read_text()).keys())
    except (json.JSONDecodeError, OSError):
        return set()


def main():
    total_combos = sum(len(b) for b in BANDS.values()) * len(YEARS)
    print(f"Total (layer, band, year) combos: {total_combos}\n")

    t_start = time.time()

    for layer_name, band_defs in BANDS.items():
        for band_name, (lo, hi) in band_defs.items():
            run_name = f"{layer_name}_{band_name}"
            t0 = time.time()
            print(f"\n{'='*80}\n{run_name}  rank band ({lo}, {hi}]\n{'='*80}", flush=True)

            tda_cfg = dict(BASE_TDA_CFG, daily_top=(lo, hi))
            path_cfg = path_cfg_for()

            for year in YEARS:
                done = existing_run_names(path_cfg["results_prefix"], year)
                if run_name in done:
                    print(f"  {year}: already present, skipping", flush=True)
                    continue
                print(f"  {year}: running", flush=True)
                run_all(
                    years=[year],
                    layers={run_name: {layer_name: LAYER_FILTERS[layer_name]}},
                    tda_cfg=tda_cfg,
                    path_cfg=path_cfg,
                )

            print(f"{run_name} DONE in {time.time()-t0:.0f}s", flush=True)

    print(f"\n\nAll (layer, band) combos complete in {time.time()-t_start:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
