"""Ricci-flow community detection pilot (idea 6 option 2 of
TDA_dev_strategy.md), on the same 2 pilot layers option 1 started with
(contract_nonFactory_ETH_only, simple_txs_ETH_only), standard
daily_top=750/max_nodes=1000. Only worth pursuing given option 1's
confirmed, layer-dependent curvature signal -- same "pilot 2 layers first,
expand only if it shows something" discipline that already worked once.

Iteratively reweights edges by Ollivier-Ricci curvature and surgically
cuts the most-negative (bridge) edges until convergence
(GraphRicciCurvature's own compute_ricci_flow, no hand-rolled surgery
logic -- the library already implements Ni/Lin/Gao/Gu/Saucan 2019's
method), then detects community partitions from the flow metric
(ricci_community). Output: how fragmented into cliques the network is
each day (count, size distribution, modularity) -- distinct from
persistent homology's loops/components and from option 1's plain
aggregate curvature scalars.

Low-RAM by construction: per-day graphs capped at max_nodes=1000, same
as every other driver in this family, nothing accumulated across days.
The cost axis here is wall-clock (compute_ricci_flow repeats the
optimal-transport curvature computation across up to `iterations`
rounds), not memory -- unlike the trailing-rank DuckDB pipeline, this
is exactly the kind of "not requiring so much RAM" work meant to run
while that pipeline waits for a clean system.

    python3 scripts/runs/run_ricci_flow_communities_pilot.py
    nohup python3 scripts/runs/run_ricci_flow_communities_pilot.py > logs/run_ricci_flow_communities_pilot.log 2>&1 &
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
import networkx.algorithms.community as nx_comm
import community.community_louvain as community_louvain
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.util import cut_graph_by_cutoff

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
FLOW_ITERATIONS = 10  # library default; delta=1e-4 (also default) early-stops most days sooner
DAY_TIMEOUT_SECONDS = 300  # generous: flow repeats the curvature computation ~FLOW_ITERATIONS
# times, vs. option 1's single call at a 120s (already-generous) budget
N_CUTOFF_STEPS = 50
DROP_THRESHOLD = 0.01


def _rf_community_scaled(G: nx.Graph, weight='weight', n_steps=N_CUTOFF_STEPS,
                          drop_threshold=DROP_THRESHOLD):
    """Reimplementation of GraphRicciCurvature's own get_rf_metric_cutoff /
    ricci_community, scaled to this project's actual edge-weight range.

    The library's version hardcodes cutoff_range = np.arange(maxw, 1,
    -cutoff_step) -- correct for its own examples, where edge weight
    defaults to 1.0 and the Ricci flow metric evolves around that. This
    pipeline instead feeds OllivierRicci the norm_similarity distance
    transform (weight in (0, 1], required for the curvature computation
    itself to mean the same thing as the already-validated option-1
    pilot), so maxw <= 1 always -- the library's hardcoded range is empty
    on every single day (confirmed directly: AssertionError "No cutoff
    point found!" on a plain smoke-test day, not a fluke). This scans
    cutoffs across the graph's own actual weight range instead, otherwise
    mirroring the library's modularity-drop logic exactly.

    Falls back to the cutoff that directly maximizes modularity when no
    clean "drop" is detected, rather than failing the day outright --
    community detection here is exploratory, and a maximum-modularity
    partition is still a real, usable answer even without a sharp drop.
    """
    weights = nx.get_edge_attributes(G, weight)
    if not weights:
        return None
    maxw, minw = max(weights.values()), min(weights.values())
    if maxw <= minw:
        return None
    cutoff_range = np.linspace(maxw, minw, n_steps)

    modularities, clusterings = [], []
    for cutoff in cutoff_range:
        Gc = cut_graph_by_cutoff(G, cutoff, weight=weight)
        clustering = {n: idx for idx, comp in enumerate(nx.connected_components(Gc)) for n in comp}
        try:
            mod = community_louvain.modularity(clustering, Gc, weight)
        except Exception:
            mod = None
        modularities.append(mod)
        clusterings.append(clustering)

    good_cut_idxs = []
    mod_last = modularities[-1]
    for i in range(len(modularities) - 1, 0, -1):
        mod_now = modularities[i]
        if mod_last is not None and mod_now is not None and mod_last > mod_now > 1e-4 \
                and abs(mod_last - mod_now) / mod_last > drop_threshold:
            good_cut_idxs.append(i + 1)
        mod_last = mod_now

    if good_cut_idxs:
        chosen_idx = good_cut_idxs[-1]  # finest-granularity detected drop, mirrors ricci_community's cc[-1]
    else:
        valid = [(i, m) for i, m in enumerate(modularities) if m is not None]
        if not valid:
            return None
        chosen_idx = max(valid, key=lambda x: x[1])[0]

    return float(cutoff_range[chosen_idx]), clusterings[chosen_idx]


MIN_COMPONENT_SIZE = 20  # below this, Ricci-flow community detection has nothing
# meaningful to subdivide -- see the fragmentation-stats fields below instead


def _ricci_flow_community_features(G: nx.Graph, max_nodes=MAX_NODES, alpha=ALPHA,
                                    iterations=FLOW_ITERATIONS,
                                    min_component_size=MIN_COMPONENT_SIZE):
    """Two distinct signals, always both reported when possible:

    1. Fragmentation stats (n_components, largest_component_frac, ...) --
       cheap (plain connected-components, no curvature/flow needed), always
       computed, and meaningful on their own: discovered directly while
       building this pilot that contract_nonFactory_ETH_only's daily
       top-750-edge graph is almost completely fragmented (largest
       component ~5-9 of ~400-500 nodes, confirmed across 4 sample days),
       while simple_txs_ETH_only has real connected structure (largest
       component 20-50% of nodes) -- a genuine, layer-dependent finding,
       not a bug. compute_ricci_flow silently restricts itself to the
       largest connected component internally (a behavior
       compute_ricci_curvature, used in option 1, does not have) -- this
       makes that restriction explicit and reports what got kept.
    2. Ricci-flow community sub-structure within the largest component --
       only attempted when that component has >= min_component_size
       nodes, since below that there's nothing for community detection to
       meaningfully subdivide (a 5-node component is either "1 community"
       or "not connected enough to ask the question," not informative
       either way).
    """
    if G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        top = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()
    if G.number_of_edges() < 2:
        return None

    all_components = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_sizes = np.array([len(c) for c in all_components], dtype=float)
    comp_probs = comp_sizes / comp_sizes.sum()
    fragmentation_stats = {
        'n_components': int(len(all_components)),
        'largest_component_size': int(comp_sizes.max()),
        'largest_component_frac': float(comp_sizes.max() / G.number_of_nodes()),
        'component_size_entropy': float(-(comp_probs * np.log(comp_probs + 1e-12)).sum()),
        'n_communities': None,
        'community_cutoff': None,
        'largest_community_frac': None,
        'community_size_std': None,
        'community_size_entropy': None,
        'modularity': None,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
    }

    if comp_sizes.max() < min_component_size:
        return fragmentation_stats

    G = G.subgraph(all_components[0]).copy()  # explicit: this is what
    # compute_ricci_flow would restrict to internally anyway if left implicit

    # Keep the original tx_count-based weights for modularity -- community
    # quality should be judged against real transaction intensity, not the
    # optimal-transport cost OllivierRicci needs as its 'weight' input.
    orig_weight = {frozenset((u, v)): d.get('weight', 0) for u, v, d in G.edges(data=True)}

    weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    A_max, A_min = max(weights), min(weights)
    weight_range = A_max - A_min
    for u, v, d in G.edges(data=True):
        w = max(d.get('weight', 1e-9), 1e-9)
        dist = 1.0 if weight_range == 0 else 1 / (1 + alpha * (w - A_min) / weight_range)
        d['weight'] = dist  # overwrite in place: OllivierRicci reads 'weight' as transport cost

    orc = OllivierRicci(G, alpha=0.5, weight='weight', verbose='ERROR')
    orc.compute_ricci_flow(iterations=iterations)
    result = _rf_community_scaled(orc.G, weight='weight')
    if result is None:
        return fragmentation_stats
    cutoff, clustering = result

    communities_by_label = {}
    for node, label in clustering.items():
        communities_by_label.setdefault(label, set()).add(node)
    communities = list(communities_by_label.values())
    sizes = np.array([len(c) for c in communities], dtype=float)
    if sizes.sum() == 0:
        return fragmentation_stats

    G_orig = G.copy()
    for u, v, d in G_orig.edges(data=True):
        d['weight'] = orig_weight.get(frozenset((u, v)), 0)
    try:
        modularity = float(nx_comm.quality.modularity(G_orig, communities, weight='weight'))
    except Exception:
        modularity = None

    probs = sizes / sizes.sum()
    fragmentation_stats.update({
        'n_communities': int(len(communities)),
        'community_cutoff': float(cutoff),
        'largest_community_frac': float(sizes.max() / sizes.sum()),
        'community_size_std': float(sizes.std()),
        'community_size_entropy': float(-(probs * np.log(probs + 1e-12)).sum()),
        'modularity': modularity,
        # n_nodes/n_edges now describe the largest-component subgraph actually
        # flowed, not the full capped graph -- overwrite deliberately
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
    })
    return fragmentation_stats


def _worker(day_df, edge_weight_col, q):
    try:
        G = _build_graph_fast(day_df, edge_weight_col=edge_weight_col)
        q.put(('ok', _ricci_flow_community_features(G)))
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
OUT = Path("results/ricci_flow_communities_pilot.parquet")


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
