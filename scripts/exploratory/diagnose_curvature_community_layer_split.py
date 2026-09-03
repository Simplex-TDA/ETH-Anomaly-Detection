"""Idea 23 (TDA_dev_strategy.md): before building a whole new node-
partitioning + daily_top + Ripser rebuild around curvature-derived
Ricci-flow communities, check cheaply whether those communities are even
(a) non-degenerate (sane count/size, not one giant blob or hundreds of
singletons) and (b) structurally *different* from the existing byte-size
split -- if curvature communities mostly just rediscover the current
nonFactory/mediumInput/highInput bands, this isn't a "replacement," it's
the same split relabeled at real engineering cost.

Same 2 representative weeks idea 12b step 5 already used and validated as
sufficient for a topology-structure question on this combined layer
(audit_byte_cutoffs_curvature.py) -- reused verbatim, not re-litigated.

For each sampled day: build the combined contract_txs_ETH_only graph (no
byte split), compute Ricci-flow communities on the largest connected
component (same method/thresholds as run_ricci_flow_communities_pilot.py),
then for every node in that component, record its curvature-community
label alongside its existing byte-band label (nonFactory/mediumInput/
highInput, from that node's own mean total_input_bytes across incident
edges that day). Cramer's V between the two labelings is the answer:
~0 means curvature communities cut across byte bands (a genuinely
different, information-bearing partition -- worth the rebuild); high
means curvature mostly re-derives the current split (not worth it).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

MAX_NODES = 1000
ALPHA = 9
DAILY_TOP = 750
FLOW_ITERATIONS = 10
N_CUTOFF_STEPS = 50
DROP_THRESHOLD = 0.01
MIN_COMPONENT_SIZE = 20

SAMPLE_WEEKS = [
    (2021, "2021-07-05", "2021-07-11"),
    (2024, "2024-07-01", "2024-07-07"),
]

t0 = time.time()


def _rf_community_scaled(G, weight='weight', n_steps=N_CUTOFF_STEPS, drop_threshold=DROP_THRESHOLD):
    """Verbatim copy of run_ricci_flow_communities_pilot.py's helper --
    see that script for the full rationale (library's own cutoff scan is
    empty at this project's weight scale)."""
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
        chosen_idx = good_cut_idxs[-1]
    else:
        valid = [(i, m) for i, m in enumerate(modularities) if m is not None]
        if not valid:
            return None
        chosen_idx = max(valid, key=lambda x: x[1])[0]

    return float(cutoff_range[chosen_idx]), clusterings[chosen_idx]


def byte_band(mean_bytes):
    if mean_bytes < 100:
        return "nonFactory"
    elif mean_bytes < 500:
        return "mediumInput"
    else:
        return "highInput"


def cramers_v(label_a, label_b):
    ct = pd.crosstab(pd.Series(label_a), pd.Series(label_b))
    chi2 = 0.0
    n = ct.values.sum()
    row_sums, col_sums = ct.values.sum(axis=1), ct.values.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(np.where(expected > 0, (ct.values - expected) ** 2 / expected, 0))
    k = min(ct.shape) - 1
    if k <= 0 or n <= 1:
        return None
    return float(np.sqrt((chi2 / n) / k))


day_rows = []
node_rows = []

for year, start, end in SAMPLE_WEEKS:
    ranking_dir = Path("data/ranking") / str(year)
    filtered_df = load_filtered_graph(
        eth_dir=Path(f"data/{year}/eth_tx_value_output/weekly"),
        erc20_dir=Path(f"data/{year}/erc20_tx_value_output/weekly"),
        global_file=ranking_dir / "global_top_nodes.parquet",
        daily_dir=ranking_dir / "daily",
        filters=["contract_txs_ETH_only"],
        ranking_metric="tx_count",
        global_top_num=0,
        daily_top_num=DAILY_TOP,
        start_date=start,
        end_date=end,
    )
    if filtered_df.empty:
        print(f"[{time.time()-t0:5.0f}s] {year} {start}..{end}: no data", flush=True)
        continue
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df = filtered_df[(filtered_df["tx_value"] == 0) & (filtered_df["erc20"] == "ETH")]
    dates = sorted(filtered_df["date"].unique())

    for d in dates:
        day_df = filtered_df[filtered_df["date"] == d]
        date_str = pd.Timestamp(d).date().isoformat()
        if len(day_df) < 3:
            print(f"[{time.time()-t0:5.0f}s] {date_str}: too small, skipping", flush=True)
            continue

        u = day_df["from_addr"].to_numpy(dtype=object)
        v = day_df["to_addr"].to_numpy(dtype=object)
        w = day_df["tx_count"].to_numpy(dtype=float)
        tib = day_df["total_input_bytes"].to_numpy(dtype=float)
        bad = (w <= 0) | np.isnan(w)
        w = np.where(bad, 1e-6, w)
        a = np.where(u <= v, u, v)
        b = np.where(u <= v, v, u)
        tmp = pd.DataFrame({"a": a, "b": b, "w": w, "tib": tib})
        summed = tmp.groupby(["a", "b"], sort=False, as_index=False).agg(w=("w", "sum"), tib=("tib", "sum"))
        G = nx.Graph()
        G.add_weighted_edges_from(zip(summed["a"], summed["b"], summed["w"]))
        for _, row in summed.iterrows():
            G[row["a"]][row["b"]]["total_input_bytes"] = row["tib"]

        if G.number_of_nodes() > MAX_NODES:
            top = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:MAX_NODES]
            G = G.subgraph([n for n, _ in top]).copy()
        if G.number_of_edges() < 2:
            print(f"[{time.time()-t0:5.0f}s] {date_str}: too few edges after cap, skipping", flush=True)
            continue

        all_components = sorted(nx.connected_components(G), key=len, reverse=True)
        comp_sizes = [len(c) for c in all_components]
        largest_frac = comp_sizes[0] / G.number_of_nodes()
        day_row = {
            "year": year, "date": date_str, "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "n_components": len(all_components), "largest_component_size": comp_sizes[0],
            "largest_component_frac": largest_frac, "n_communities": None, "modularity": None,
            "cramers_v_community_vs_byteband": None,
        }

        if comp_sizes[0] < MIN_COMPONENT_SIZE:
            print(f"[{time.time()-t0:5.0f}s] {date_str}: largest component {comp_sizes[0]} < {MIN_COMPONENT_SIZE}, "
                  f"fragmentation only", flush=True)
            day_rows.append(day_row)
            continue

        Gc = G.subgraph(all_components[0]).copy()
        orig_edges = {frozenset((u_, v_)): d for u_, v_, d in Gc.edges(data=True)}

        weights = [d.get("weight", 0) for _, _, d in Gc.edges(data=True)]
        A_max, A_min = max(weights), min(weights)
        weight_range = A_max - A_min
        for u_, v_, d in Gc.edges(data=True):
            wt = max(d.get("weight", 1e-9), 1e-9)
            d["weight"] = 1.0 if weight_range == 0 else 1 / (1 + ALPHA * (wt - A_min) / weight_range)

        orc = OllivierRicci(Gc, alpha=0.5, weight="weight", verbose="ERROR")
        orc.compute_ricci_flow(iterations=FLOW_ITERATIONS)
        result = _rf_community_scaled(orc.G, weight="weight")
        if result is None:
            print(f"[{time.time()-t0:5.0f}s] {date_str}: community detection returned None", flush=True)
            day_rows.append(day_row)
            continue
        cutoff, clustering = result

        communities_by_label = {}
        for node, label in clustering.items():
            communities_by_label.setdefault(label, set()).add(node)
        communities = list(communities_by_label.values())
        try:
            modularity = float(nx_comm.quality.modularity(
                Gc, communities, weight="weight"))  # NB: Gc weights already overwritten to distance here;
            # fine for this diagnostic's purpose (relative modularity sanity check), unlike the pilot's
            # more careful original-weight restoration -- not reused downstream as a feature.
        except Exception:
            modularity = None

        # per-node mean total_input_bytes across this day's incident edges (from the original,
        # uncapped graph's edge attributes where available, falling back to the capped subgraph)
        node_bytes = {}
        for node in Gc.nodes():
            incident = [orig_edges[k]["total_input_bytes"] for k in orig_edges if node in k]
            node_bytes[node] = float(np.mean(incident)) if incident else np.nan

        community_labels = [clustering[n] for n in Gc.nodes()]
        byte_bands = [byte_band(node_bytes[n]) for n in Gc.nodes()]
        cv = cramers_v(community_labels, byte_bands)

        day_row.update({"n_communities": len(communities), "modularity": modularity, "cramers_v_community_vs_byteband": cv})
        day_rows.append(day_row)

        for n in Gc.nodes():
            node_rows.append({"year": year, "date": date_str, "node": n, "community": clustering[n],
                               "mean_total_input_bytes": node_bytes[n], "byte_band": byte_band(node_bytes[n])})

        print(f"[{time.time()-t0:5.0f}s] {date_str}: {G.number_of_nodes()} nodes, largest_component_frac={largest_frac:.2f}, "
              f"{len(communities)} communities, modularity={modularity}, Cramer's V vs byte-band={cv}", flush=True)

day_df_out = pd.DataFrame(day_rows)
node_df_out = pd.DataFrame(node_rows)
day_df_out.to_parquet("results/exploratory/curvature_community_layer_split_days.parquet")
node_df_out.to_parquet("results/exploratory/curvature_community_layer_split_nodes.parquet")

print(f"\n[{time.time()-t0:5.0f}s] {len(day_df_out)} days processed, {len(node_df_out)} node-day records", flush=True)
print("\n=== Day-level summary ===")
print(day_df_out.to_string(index=False))
valid_cv = day_df_out["cramers_v_community_vs_byteband"].dropna()
if len(valid_cv):
    print(f"\nCramer's V (community label vs. byte-band label), across {len(valid_cv)} days with community detection:")
    print(f"  mean={valid_cv.mean():.3f}  median={valid_cv.median():.3f}  min={valid_cv.min():.3f}  max={valid_cv.max():.3f}")
    print("  (0 = communities cut across byte bands, genuinely different structure;"
          " 1 = communities perfectly re-derive the existing byte split)")
else:
    print("\nNo days had community detection succeed -- see largest_component_frac / n_components above.")

print("\n\nDONE.")
