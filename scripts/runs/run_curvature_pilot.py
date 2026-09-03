"""Ollivier-Ricci curvature pilot -- daily aggregate scalars only (option 1
of TDA_dev_strategy.md idea 6), on the 2 pilot layers (contract_
nonFactory_ETH_only, simple_txs_ETH_only), standard daily_top=750/
max_nodes=1000 (the project's established default -- this isn't a
daily_top-sensitivity test, it's a new parallel feature family, so using
the most-comparable existing config rather than the sweep's raised caps).

No Ripser call at all: same load_filtered_graph + build_graph as every
other driver in this family, but the graph goes straight into
GraphRicciCurvature's OllivierRicci instead of geodesic_densification +
Ripser. Same norm_similarity weight->distance transform as
geodesic_densification (alpha=9) for consistency with the rest of the
pipeline, applied as the curvature library's edge 'weight' (its
optimal-transport cost), and the same degree-based max_nodes capping.

Output: one row per (layer, date) with 5 aggregate curvature scalars --
plugs directly into the existing Arm-I dynamics pipeline the same way
any H0_*/H1_* scalar does today.

    python3 scripts/runs/run_curvature_pilot.py
    nohup python3 scripts/runs/run_curvature_pilot.py > logs/run_curvature_pilot.log 2>&1 &
"""
import json
import multiprocessing as mp
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
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import tad_ethereum_functions as tef
from tad_ethereum_functions import load_filtered_graph


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


MAX_NODES = 1000
ALPHA = 9
DAY_TIMEOUT_SECONDS = 120  # generous given curvature has no Ripser step; defensive only


def _curvature_scalars(G: nx.Graph, max_nodes=MAX_NODES, alpha=ALPHA):
    if G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        top = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()
    if G.number_of_edges() < 2:
        return None

    weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    A_max, A_min = max(weights), min(weights)
    weight_range = A_max - A_min
    for u, v, d in G.edges(data=True):
        w = max(d.get('weight', 1e-9), 1e-9)
        if weight_range == 0:
            dist = 1.0
        else:
            dist = 1 / (1 + alpha * (w - A_min) / weight_range)
        d['weight'] = dist  # overwrite in place: OllivierRicci reads 'weight' as transport cost

    orc = OllivierRicci(G, alpha=0.5, weight='weight', verbose='ERROR')
    orc.compute_ricci_curvature()
    vals = np.array([d['ricciCurvature'] for _, _, d in orc.G.edges(data=True)])
    if len(vals) == 0:
        return None

    edge_weights = np.array([1.0 / max(d.get('weight', 1e-9), 1e-9) for _, _, d in orc.G.edges(data=True)])
    return {
        'curvature_mean': float(vals.mean()),
        'curvature_median': float(np.median(vals)),
        'curvature_min': float(vals.min()),
        'curvature_frac_negative': float((vals < 0).mean()),
        'curvature_weighted_mean': float(np.average(vals, weights=edge_weights)),
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
    }


def _worker(day_df, edge_weight_col, q):
    try:
        G = _build_graph_fast(day_df, edge_weight_col=edge_weight_col)
        q.put(('ok', _curvature_scalars(G)))
    except Exception as e:
        q.put(('error', repr(e)))


def curvature_for_day(day_df, edge_weight_col='tx_count'):
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(day_df, edge_weight_col, q))
    p.start()
    p.join(DAY_TIMEOUT_SECONDS)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        q.close()
        return None, 'timeout'
    if not q.empty():
        status, payload = q.get()
        p.join()
        q.close()
        return (payload, None) if status == 'ok' else (None, payload)
    p.join()
    q.close()
    return None, f'died (exitcode={p.exitcode})'


# layer_key -> (filter_fn, ranking_metric, edge_weight_col, base_filter_name).
# base_filter_name is the underlying ranking-file filter (only differs from
# the layer_key for the tx_value-ranked simple_txs variant, matching the
# daily_top sweep's naming convention).
LAYER_FILTERS = {
    "simple_txs_ETH_only":           (lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "simple_txs_ETH_only"),
    "contract_nonFactory_ETH_only":  (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_nonFactory_ETH_only"),
    "contract_mediumInput_ETH_only": (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_mediumInput_ETH_only"),
    "contract_highInput_ETH_only":   (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_highInput_ETH_only"),
    "simple_txs_ETH_only_value":     (lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"), "tx_value", "tx_value", "simple_txs_ETH_only"),
}
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAILY_TOP = 750
OUT = Path("results/curvature_pilot_results.parquet")


def existing_results():
    if OUT.exists():
        return pd.read_parquet(OUT)
    return pd.DataFrame(columns=['layer', 'date'])


def main():
    t0 = time.time()
    done = existing_results()
    all_rows = [done] if not done.empty else []

    for layer_name, (filter_fn, ranking_metric, edge_weight_col, base_filter_name) in LAYER_FILTERS.items():
        for year in YEARS:
            done_dates = set(done[(done['layer'] == layer_name)]['date']) if not done.empty else set()

            ranking_dir = Path("data/ranking") / str(year)
            filtered_df = load_filtered_graph(
                eth_dir=Path(f"data/{year}/eth_tx_value_output/weekly"),
                erc20_dir=Path(f"data/{year}/erc20_tx_value_output/weekly"),
                global_file=ranking_dir / 'global_top_nodes.parquet',
                daily_dir=ranking_dir / 'daily',
                filters=[base_filter_name],
                ranking_metric=ranking_metric,
                global_top_num=0,
                daily_top_num=DAILY_TOP,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
            )
            if filtered_df.empty:
                print(f"[{time.time()-t0:6.0f}s] {layer_name} {year}: no data", flush=True)
                continue
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            dates = sorted(filtered_df['date'].unique())

            rows = []
            for d in dates:
                date_str = pd.Timestamp(d).date().isoformat()
                if date_str in done_dates:
                    continue
                day_df = filtered_df[filtered_df['date'] == d]
                day_df = day_df[filter_fn(day_df)]  # load_filtered_graph only restricts by node
                                                     # ranking membership -- the layer's actual
                                                     # transaction-type condition (e.g. tx_value==0
                                                     # for contract layers) is applied here, matching
                                                     # compute_day_pd's semantics.
                if len(day_df) < 3:
                    continue
                result, err = curvature_for_day(day_df, edge_weight_col=edge_weight_col)
                if err is not None:
                    print(f"  [SKIP] {layer_name} {date_str}: {err}", flush=True)
                    continue
                if result is None:
                    continue
                result['layer'] = layer_name
                result['date'] = date_str
                rows.append(result)

            if rows:
                all_rows.append(pd.DataFrame(rows))
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_parquet(OUT)
                all_rows = [combined]
            print(f"[{time.time()-t0:6.0f}s] {layer_name} {year}: {len(rows)} new days computed "
                  f"({len(dates)} total dates in filtered data)", flush=True)

    print(f"\n[{time.time()-t0:6.0f}s] DONE.", flush=True)


if __name__ == "__main__":
    main()
