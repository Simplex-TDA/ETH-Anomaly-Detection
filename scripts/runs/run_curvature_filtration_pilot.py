"""Curvature-as-filtration persistent homology pilot (idea 6 option 3b of
TDA_dev_strategy.md), on the same 2 pilot layers option 1 started with
(contract_nonFactory_ETH_only, simple_txs_ETH_only), standard
daily_top=750/max_nodes=1000. Only worth pursuing given option 1's
confirmed, layer-dependent curvature signal.

A genuinely different diagram from the production pipeline's, not a
variant of it: instead of Vietoris-Rips on geodesic distance from
norm_similarity (`geodesic_densification` + `compute_pd`), this builds a
flag complex directly from raw adjacency and assigns each EDGE's
filtration value from its own Ollivier-Ricci curvature (single pass,
`compute_ricci_curvature()` -- no Ricci flow, unlike option 2's community
pilot). Birth/death under this filtration tracks bridge-vs-cluster
structure directly rather than proximity.

Sign convention (this pilot's own design choice, not GUDHI's): filtration
= -curvature. Cluster-interior edges (high +curvature, "redundant"
connections) get very negative filtration values and enter first, mirroring
how strong ties enter early under a geodesic-distance filtration; bridge
edges (negative curvature, structurally critical connectors) get positive
filtration values and enter last, mirroring how weak/long-distance ties
enter late. This is the same "cut the most-negative edges last, they're
structurally load-bearing" intuition idea 6's Ricci-flow community
detection (option 2) already uses, just expressed as a filtration order
instead of a surgery order.

No plumbing reuse from the production Ripser pipeline was possible:
`persistence_features()` (tad_ethereum_functions.py:394) is hard-coded to
call `ripser()` internally on a distance matrix, not accept a precomputed
diagram -- confirmed by reading it directly. This reimplements its exact
same 9-scalar formulas (verbatim, same variable names) against a
GUDHI-native diagram instead, converted into ripser's [H0_array, H1_array]
shape first so the two are computed identically and stay comparable.

    python3 scripts/runs/run_curvature_filtration_pilot.py
    nohup python3 scripts/runs/run_curvature_filtration_pilot.py > logs/run_curvature_filtration_pilot.log 2>&1 &
"""
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
import gudhi
from scipy.stats import entropy
from GraphRicciCurvature.OllivierRicci import OllivierRicci

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
ALPHA = 9  # norm_similarity distance-transform constant, same as every other driver -- NOT
# OllivierRicci's own alpha=0.5 mass parameter passed to the OllivierRicci constructor below
DAY_TIMEOUT_SECONDS = 180  # curvature (single pass, no flow) + GUDHI expansion/persistence;
# generous relative to option 1's 120s given the extra GUDHI step, still no flow-iteration cost


def _persistence_features_from_dgms(diagrams):
    """Verbatim reimplementation of tad_ethereum_functions.persistence_features's
    9-scalar formulas (same variable names, same order), taking an already-
    computed [H0_array, H1_array] diagram pair instead of calling ripser()
    internally -- see this module's docstring for why persistence_features
    itself couldn't be reused directly."""
    features = {}

    H0 = diagrams[0]
    H0_lifetimes = H0[:, 1] - H0[:, 0]
    H0_lifetimes = H0_lifetimes[np.isfinite(H0_lifetimes)]

    if len(diagrams) > 1:
        H1 = diagrams[1]
        H1_lifetimes = H1[:, 1] - H1[:, 0]
        H1_lifetimes = H1_lifetimes[np.isfinite(H1_lifetimes)]
    else:
        H1_lifetimes = np.array([])

    features['num_loops'] = len(H1_lifetimes)

    all_lifetimes = np.concatenate([H0_lifetimes, H1_lifetimes]) if len(H1_lifetimes) > 0 else H0_lifetimes
    features['avg_persistence'] = np.mean(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
    features['max_persistence'] = np.max(all_lifetimes) if len(all_lifetimes) > 0 else 0.0

    if len(all_lifetimes) > 0:
        probs = all_lifetimes / (np.sum(all_lifetimes) + 1e-10)
        features['persistence_entropy'] = entropy(probs)
    else:
        features['persistence_entropy'] = 0.0

    features['avg_loop_persistence'] = np.mean(H1_lifetimes) if len(H1_lifetimes) > 0 else 0.0
    features['max_loop_persistence'] = np.max(H1_lifetimes) if len(H1_lifetimes) > 0 else 0.0

    total_persistence = np.sum(all_lifetimes) + 1e-10
    loop_persistence = np.sum(H1_lifetimes)
    features['loop_persistence_ratio'] = loop_persistence / total_persistence

    if len(H1_lifetimes) > 0:
        threshold = np.percentile(H1_lifetimes, 75)
        features['num_strong_loops'] = int(np.sum(H1_lifetimes > threshold))
    else:
        features['num_strong_loops'] = 0

    features['persistence_variance'] = np.var(all_lifetimes) if len(all_lifetimes) > 0 else 0.0

    return features


def _curvature_filtration_features(G: nx.Graph, max_nodes=MAX_NODES, alpha=ALPHA):
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
        dist = 1.0 if weight_range == 0 else 1 / (1 + alpha * (w - A_min) / weight_range)
        d['weight'] = dist  # overwrite in place: OllivierRicci reads 'weight' as transport cost

    orc = OllivierRicci(G, alpha=0.5, weight='weight', verbose='ERROR')
    orc.compute_ricci_curvature()

    idx = {node: i for i, node in enumerate(orc.G.nodes())}
    edge_filtrations = {(idx[u], idx[v]): -float(d.get('ricciCurvature', 0.0))
                         for u, v, d in orc.G.edges(data=True)}
    min_filtration = min(edge_filtrations.values())

    st = gudhi.SimplexTree()
    # Insert every vertex explicitly at the graph's global-minimum filtration
    # BEFORE any edge -- matching ripser's own convention (H0 birth is always
    # exactly 0 in every diagram it produces). Without this, GUDHI defaults a
    # newly-created vertex's filtration to whatever simplex introduced it, so
    # any small/isolated component (this project's contract-call layers are
    # extremely fragmented -- see idea 6 option 2's finding of ~400 separate
    # 2-3 node components per day) gets its vertices born at the SAME
    # filtration as their one connecting edge, giving a spurious 0-length H0
    # lifetime for nearly every point -- confirmed directly: avg_persistence
    # came out exactly 0.0 (entropy nan) for contract_nonFactory_ETH_only
    # before this fix, vs. real non-zero values after.
    for node in orc.G.nodes():
        st.insert([idx[node]], filtration=min_filtration)
    for (i, j), f in edge_filtrations.items():
        st.insert([i, j], filtration=f)
    st.expansion(2)  # build 2-simplices so H1 boundary maps are well-formed, same reasoning
    # Ripser's maxdim=1 uses internally (needs the next dimension up)
    st.compute_persistence()

    h0 = st.persistence_intervals_in_dimension(0)
    h1 = st.persistence_intervals_in_dimension(1)
    if len(h0) == 0:
        h0 = np.array([[0., 0.]])
    if len(h1) == 0:
        h1 = np.array([[0., 0.]])
    diagrams = [np.asarray(h0), np.asarray(h1)]

    result = _persistence_features_from_dgms(diagrams)
    result['n_nodes'] = G.number_of_nodes()
    result['n_edges'] = G.number_of_edges()
    return result


def _worker(day_df, edge_weight_col, q):
    try:
        G = _build_graph_fast(day_df, edge_weight_col=edge_weight_col)
        q.put(('ok', _curvature_filtration_features(G)))
    except Exception as e:
        q.put(('error', repr(e)))


def features_for_day(day_df, edge_weight_col='tx_count'):
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


# All 5 layers used by option 1's curvature pilot (run_curvature_pilot.py's
# LAYER_FILTERS, verbatim) -- expanded from the original 2-layer pilot scope
# once option 1's per-layer signal justified going wider.
LAYER_FILTERS = {
    "simple_txs_ETH_only":           (lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "simple_txs_ETH_only"),
    "contract_nonFactory_ETH_only":  (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] < 100) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_nonFactory_ETH_only"),
    "contract_mediumInput_ETH_only": (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_mediumInput_ETH_only"),
    "contract_highInput_ETH_only":   (lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 500) & (d["erc20"] == "ETH"), "tx_count", "tx_count", "contract_highInput_ETH_only"),
    "simple_txs_ETH_only_value":     (lambda d: (d["tx_value"] >  0) & (d["erc20"] == "ETH"), "tx_value", "tx_value", "simple_txs_ETH_only"),
}
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAILY_TOP = 750
OUT = Path("results/curvature_filtration_pilot.parquet")


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
                day_df = day_df[filter_fn(day_df)]
                if len(day_df) < 3:
                    continue
                result, err = features_for_day(day_df, edge_weight_col=edge_weight_col)
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
