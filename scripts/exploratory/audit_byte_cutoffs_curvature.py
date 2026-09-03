"""Idea 12b, step 5: does the current 100/500-byte split land on natural
bridges (low/negative Ollivier-Ricci curvature) or does it arbitrarily
bisect cluster interiors (high curvature)? Build the COMBINED
contract_txs_ETH_only graph (no byte split -- already-ranked filter,
idea 21) for a couple of representative weeks, compute per-edge
curvature, and bin edges by their own total_input_bytes to see whether
mean curvature dips near 100/500 (bridge-like, a threshold the graph's
own topology would have chosen anyway) or is flat/rising through those
points (arbitrarily bisecting real structure).

Scoped to 2 representative weeks (not the full 6-year sweep) to keep
`load_filtered_graph`'s cost bounded -- this is a confirmatory check on
top of steps 1+3's already-strong evidence, not a full rebuild.
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
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from tad_ethereum_functions import load_filtered_graph

MAX_NODES = 1000
ALPHA = 9
DAILY_TOP = 750

SAMPLE_WEEKS = [
    (2021, "2021-07-05", "2021-07-11"),
    (2024, "2024-07-01", "2024-07-07"),
]

t0 = time.time()


def build_graph_with_bytes(day_df):
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
    for _, row in summed.iterrows():
        G.add_edge(row["a"], row["b"], weight=row["w"], total_input_bytes=row["tib"])
    return G


def curvature_with_bytes(G, max_nodes=MAX_NODES, alpha=ALPHA):
    if G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        top = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()
    if G.number_of_edges() < 2:
        return None

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    A_max, A_min = max(weights), min(weights)
    weight_range = A_max - A_min
    for u, v, d in G.edges(data=True):
        w = max(d.get("weight", 1e-9), 1e-9)
        d["dist"] = 1.0 if weight_range == 0 else 1 / (1 + alpha * (w - A_min) / weight_range)

    G2 = nx.Graph()
    for u, v, d in G.edges(data=True):
        G2.add_edge(u, v, weight=d["dist"], total_input_bytes=d["total_input_bytes"])

    orc = OllivierRicci(G2, alpha=0.5, weight="weight", verbose="ERROR")
    orc.compute_ricci_curvature()
    rows = [{"total_input_bytes": d["total_input_bytes"], "curvature": d["ricciCurvature"]}
            for _, _, d in orc.G.edges(data=True)]
    return pd.DataFrame(rows)


all_rows = []
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
    filtered_df = filtered_df[(filtered_df["tx_value"] == 0) & (filtered_df["erc20"] == "ETH")]  # contract_txs_ETH_only, no byte split
    dates = sorted(filtered_df["date"].unique())
    for d in dates:
        day_df = filtered_df[filtered_df["date"] == d]
        if len(day_df) < 3:
            continue
        G = build_graph_with_bytes(day_df)
        result = curvature_with_bytes(G)
        if result is not None:
            result["year"] = year
            result["date"] = pd.Timestamp(d).date().isoformat()
            all_rows.append(result)
        print(f"[{time.time()-t0:5.0f}s] {year}-{pd.Timestamp(d).date()}: "
              f"{'skipped (too small)' if result is None else f'{len(result)} edges'}", flush=True)

edge_df = pd.concat(all_rows, ignore_index=True)
edge_df.to_parquet("results/exploratory/byte_cutoff_curvature.parquet")
print(f"\n[{time.time()-t0:5.0f}s] Total edges with curvature+bytes: {len(edge_df):,}", flush=True)

print("\n=== mean/median curvature by total_input_bytes bin (0-1000, 25-byte bins) ===")
mask = (edge_df["total_input_bytes"] > 0) & (edge_df["total_input_bytes"] <= 1000)
sub = edge_df[mask].copy()
bins = np.arange(0, 1025, 25)
sub["bin"] = pd.cut(sub["total_input_bytes"], bins)
agg = sub.groupby("bin", observed=True)["curvature"].agg(["mean", "median", "count"])
for interval, row in agg.iterrows():
    marker = "  <-- 100" if interval.left <= 100 < interval.right else ("  <-- 500" if interval.left <= 500 < interval.right else "")
    print(f"  {str(interval):20s} mean={row['mean']:+.4f} median={row['median']:+.4f} n={int(row['count']):6d}{marker}")

print("\n=== overall stats ===")
print(edge_df["curvature"].describe())
print(f"\nPearson corr(total_input_bytes, curvature), full range: {edge_df['total_input_bytes'].corr(edge_df['curvature']):.4f}")
print(f"Pearson corr(total_input_bytes, curvature), 0-1000 only: {sub['total_input_bytes'].corr(sub['curvature']):.4f}")

print("\n\nDONE.")
