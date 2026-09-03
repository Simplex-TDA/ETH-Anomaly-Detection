"""Pilot: rank nodes by trailing-7-day summed tx_count instead of just
today's tx_count, keeping everything else identical to the standard
pipeline -- same daily_top=750 (edges, not nodes -- matching rank_nodes'
existing edge-based convention), same max_nodes=1000, same day's own
edges only (this is a node-SELECTION-criterion test, not idea 13's
whole-graph aggregation). One layer (contract_nonFactory_ETH_only, the
strongest/most-characterized layer), full 2020-2025.

Switched from a 30-day to a 7-day window after the 30-day version
self-aborted at 6GB RSS partway through even a single year (2020),
across three different memory strategies (string interning, flat-tuple
accumulation, integer-encoded addresses) that all landed in the same
range -- the ceiling was real data cardinality for this layer's 30-day
active-pair population, not a fixable inefficiency. `pair_sum`'s size is
roughly proportional to window length, so 7 days needs ~7/30 (~23%) of
the 30-day version's memory -- plausibly enough headroom to complete the
full 6 years without switching to a smaller layer.

Streaming design: process one weekly file at a time, feed its days into
the incremental ranking loop immediately, then drop the file's raw data.
Peak memory is bounded by one file plus the trailing-window's pair_sum
dict, not the full 6-year history.

Incremental sliding-window ranking: maintain a running per-(from_addr,
to_addr)-pair trailing-sum as a dict, add each new day's contribution,
evict days that fall outside the window, re-rank top-750 pairs each day.

Downstream: run_tda_pipeline + persistence_features (the actual
production functions, not reimplementations) give 9 raw scalars/day,
saved for regime_detection's dynamics-treatment + confirmatory pipeline.
"""
import glob
import gc
import heapq
import os
import resource
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

MAX_RSS_KB = 9_000_000  # ~9GB self-abort threshold -- real margin below the 57GB-swap
# incident, but enough headroom for a genuine full-6-year attempt at 7-day window sizing

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, "functions")

import pandas as pd

from tad_ethereum_functions import run_tda_pipeline

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]  # back to full scope -- 7-day window should have
# enough memory headroom to make this feasible where 30-day wasn't
WINDOW_DAYS = 7
TOP_N_EDGES = 750
MAX_NODES = 1000
OUT_RAW = Path("results/exploratory/trailing_rank_pilot_raw.parquet")

# Bounded-size approximation for pair_sum, added after the 2024 activity spike
# (pair_sum went 1.36M -> 5.57M -> 14.2M across two 50-day windows, blowing past
# 9GB): once pair_sum exceeds this cap, keep only the largest-value entries.
# Only the largest entries can ever matter for TOP_N_EDGES=750 selection, and
# the cap is ~2700x that, so this shouldn't change which pairs make top-750 in
# practice -- the only real precision loss is a currently-small pair that later
# spikes hard enough within its remaining window membership to reach top-750,
# which would restart its accumulation from the day it re-enters pair_sum
# rather than its true window-start. Rare, bounded, and the right tradeoff for
# a pilot meant to validate the mechanism cheaply -- a full-fidelity version
# would want a proper streaming top-K structure (or more memory) instead.
MAX_PAIR_SUM_SIZE = 2_000_000

t0 = time.time()

pair_sum = defaultdict(float)
day_queue = deque()  # (date, {(from_id,to_id): tx_count})
qualifying_rows = []  # flat list of (date, from_id, to_id, tx_count) tuples -- NOT one DataFrame per day
n_days_processed = 0

# Integer-encoding addresses (tried, reverted) made growth *worse*: addr_to_id
# never stopped discovering brand-new addresses even within 100 days, so it
# added a second growing structure on top of everything else.
#
# sys.intern() (tried, reverted) hit a subtler version of the same problem:
# even after capping pair_sum, RSS jumped another ~1.4GB going into 2024's
# fresh wave of never-seen addresses, because CPython's intern table is
# permanent and unboundable -- once interned, a string stays cached for the
# process's entire life regardless of whether pair_sum still references it.
# Pruning pair_sum can't touch it. This own_intern cache replaces sys.intern
# with the same deduplication benefit but one we can actually bound: clear
# it outright once it exceeds the cap. Safe to do at any time -- clearing
# only affects *future* dedup effectiveness, not already-referenced strings
# (those stay alive via pair_sum/day_queue/qualifying_rows regardless).
own_intern = {}
MAX_INTERN_SIZE = 2_000_000


def dedupe(s):
    cached = own_intern.get(s)
    if cached is not None:
        return cached
    if len(own_intern) >= MAX_INTERN_SIZE:
        own_intern.clear()
    own_intern[s] = s
    return s


# Second-order problem, found after own_intern still hit the same 9GB wall
# at the same day (1600) as pair_sum-pruning alone: qualifying_rows retains
# EVERY row for the whole 6-year run (it's the actual output, can't be
# pruned) -- and each time own_intern clears, addresses that keep
# reappearing in qualifying_rows across clear-cycles get a fresh, separate
# string object each time, so qualifying_rows itself accumulates duplicate
# copies of the same address content. Fix: a second cache, never cleared,
# but scoped ONLY to addresses that actually make it into qualifying_rows
# (top-750-edge endpoints per day) -- a much smaller population than every
# pair pair_sum ever considered (which includes the vast one-off long tail
# that's exactly what made the original sys.intern() unbounded).
output_intern = {}


def output_dedupe(s):
    cached = output_intern.get(s)
    if cached is not None:
        return cached
    output_intern[s] = s
    return s


def process_day(date, day_df):
    global n_days_processed, pair_sum
    day_pairs = {(dedupe(r.from_addr), dedupe(r.to_addr)): r.tx_count for r in day_df.itertuples(index=False)}

    for pair, val in day_pairs.items():
        pair_sum[pair] += val
    day_queue.append((date, day_pairs))

    while day_queue and (date - day_queue[0][0]).days >= WINDOW_DAYS:
        old_date, old_pairs = day_queue.popleft()
        for pair, val in old_pairs.items():
            pair_sum[pair] -= val
            if pair_sum[pair] <= 1e-9:
                del pair_sum[pair]

    top_pairs = heapq.nlargest(TOP_N_EDGES, pair_sum.items(), key=lambda kv: kv[1])
    node_set = set()
    for (a, b), _ in top_pairs:
        node_set.add(a)
        node_set.add(b)

    if len(pair_sum) > MAX_PAIR_SUM_SIZE:
        kept = heapq.nlargest(MAX_PAIR_SUM_SIZE, pair_sum.items(), key=lambda kv: kv[1])
        pair_sum = defaultdict(float, kept)

    # build qualifying rows straight from day_pairs (already-interned keys) --
    # NOT a pandas boolean-mask slice of day_df, which was the actual memory
    # driver: a full new DataFrame object (Index + BlockManager + dtype
    # overhead) created once per day, ~2200 times, cost ~10MB/day regardless
    # of row count. A flat tuple list defers all pandas overhead to one
    # single DataFrame() call at the very end.
    for (a, b), val in day_pairs.items():
        if a in node_set and b in node_set:
            qualifying_rows.append((date, output_dedupe(a), output_dedupe(b), val))

    n_days_processed += 1
    if n_days_processed % 50 == 0:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024  # macOS reports bytes, not KB, from getrusage
        print(f"[{time.time()-t0:6.0f}s] {n_days_processed} days processed "
              f"(pair_sum size={len(pair_sum):,}, own_intern size={len(own_intern):,}, "
              f"output_intern size={len(output_intern):,}, qualifying_rows={len(qualifying_rows):,}, "
              f"rss={rss_kb:,}KB, last date={date.date()})", flush=True)
        if rss_kb > MAX_RSS_KB:
            print(f"ABORT: rss {rss_kb:,}KB exceeds {MAX_RSS_KB:,}KB safety threshold -- "
                  f"stopping before this risks another swap blowup.", flush=True)
            sys.exit(1)


for year in YEARS:
    files = sorted(glob.glob(f"data/{year}/eth_tx_value_output/weekly/eth_tx_*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["date", "from_addr", "to_addr", "tx_value", "total_input_bytes", "tx_count"])
        df = df[(df["tx_value"] == 0) & (df["total_input_bytes"] < 100)]
        df = df[["date", "from_addr", "to_addr", "tx_count"]]
        df["date"] = pd.to_datetime(df["date"])
        for date, day_df in df.groupby("date", sort=True):
            process_day(date, day_df)
        del df
        gc.collect()
    print(f"[{time.time()-t0:6.0f}s] year {year}: done ({n_days_processed} cumulative days)", flush=True)

filtered_df = pd.DataFrame(qualifying_rows, columns=["date", "from_addr", "to_addr", "tx_count"])
print(f"[{time.time()-t0:6.0f}s] Trailing-window-filtered edge-days: {len(filtered_df):,} "
      f"(avg {len(filtered_df)/n_days_processed:.1f} edges/day)", flush=True)
del qualifying_rows
gc.collect()

# ── run the actual production Ripser pipeline, single-layer mode ──
pd_series, features_series, stats_df = run_tda_pipeline(
    filtered_df,
    edge_weight_col="tx_count",
    layer_filters=None,
    max_nodes=MAX_NODES,
    maxdim=1,
    similarity_metric="norm_similarity",
    alpha=9,
)

rows = []
for date, feats in features_series.items():
    rows.append({"date": date, **feats})
result = pd.DataFrame(rows).sort_values("date")
result.to_parquet(OUT_RAW)
print(f"\n[{time.time()-t0:6.0f}s] Saved {len(result)} days of raw scalars -> {OUT_RAW}", flush=True)
print(result.describe())

print("\n\nDONE.")
