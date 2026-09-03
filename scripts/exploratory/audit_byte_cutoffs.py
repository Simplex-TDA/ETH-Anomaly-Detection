"""Idea 12b, steps 1 + a correlation check: is the current
100/500-byte `total_input_bytes` cutoff (contract_nonFactory <100 /
contract_mediumInput 100-500 / contract_highInput >=500) landing on real
breakpoints in the distribution, or on an arbitrary point?

Sampling follows graph_prep.md's algorithm: representative weekly files
(~1 week every 2 months, all 6 years) loaded per-batch, columns projected
down before materializing, contract-call rows only (tx_value == 0).

A wrinkle worth flagging before interpreting the histogram: `total_input_
bytes` is SUM(n_input_nonzero_bytes) over every transaction for a given
(from_addr, to_addr, date) edge (confirmed in eth_data_fetcher.py's
ClickHouse query), not a single call's calldata size. An edge with 10
simple 10-byte calls and an edge with one 100-byte call both land at
total_input_bytes=100 under the current filter. This script checks that
conflation directly (correlation with tx_count) alongside the raw
distribution shape, since it changes what "the cutoff" is actually
splitting on. `min_input_bytes` (a true per-transaction, per-edge-day
minimum) is also fetched and reported for contrast.

Output: printed distribution/correlation report + saved histogram data
(no plotting library assumed available; bin counts printed as a text
histogram).
"""
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = ["01", "03", "05", "07", "09", "11"]
COLS = ["date", "tx_value", "tx_count", "total_input_bytes", "min_input_bytes"]

t0 = time.time()
frames = []
for year in YEARS:
    for month in MONTHS:
        pattern = f"data/{year}/eth_tx_value_output/weekly/eth_tx_*_{year}-{month}_W1.parquet"
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        f = matches[0]
        tbl = pq.read_table(f, columns=COLS)
        df = tbl.to_pandas()
        # contract-call edges only (tx_value == 0), matching the layer definition
        df = df[df["tx_value"] == 0]
        frames.append(df[["total_input_bytes", "tx_count", "min_input_bytes"]])
        print(f"[{time.time()-t0:5.0f}s] {f}: {len(df):,} contract-call edge-days", flush=True)

all_df = pd.concat(frames, ignore_index=True)
print(f"\n[{time.time()-t0:5.0f}s] Total sampled contract-call edge-days: {len(all_df):,}", flush=True)
all_df.to_parquet("results/exploratory/byte_cutoff_sample.parquet")

tib = all_df["total_input_bytes"]
print("\n=== total_input_bytes (SUM per edge-day) distribution, contract-call edges ===")
print(tib.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== zero/near-zero mass ===")
print(f"  == 0:        {(tib == 0).mean():.4f}")
print(f"  in (0,100):  {((tib > 0) & (tib < 100)).mean():.4f}")
print(f"  in [100,500):{((tib >= 100) & (tib < 500)).mean():.4f}")
print(f"  >= 500:      {(tib >= 500).mean():.4f}")

print("\n=== text histogram, 0-1000 bytes (where the cutoffs live), 20-byte bins ===")
mask = (tib > 0) & (tib <= 1000)
bins = np.arange(0, 1020, 20)
hist, edges = np.histogram(tib[mask], bins=bins)
maxh = hist.max()
for i in range(len(hist)):
    bar = "#" * int(60 * hist[i] / maxh)
    marker = "  <-- 100" if edges[i] <= 100 < edges[i + 1] else ("  <-- 500" if edges[i] <= 500 < edges[i + 1] else "")
    print(f"  [{edges[i]:4.0f},{edges[i+1]:4.0f}) {hist[i]:8d} {bar}{marker}")

print("\n=== log-scale histogram, full range, to see the long tail's shape ===")
pos = tib[tib > 0]
log_bins = np.logspace(0, np.log10(pos.max()), 40)
hist, edges = np.histogram(pos, bins=log_bins)
maxh = hist.max()
for i in range(len(hist)):
    bar = "#" * int(60 * hist[i] / maxh) if maxh > 0 else ""
    marker = ""
    if edges[i] <= 100 < edges[i + 1]:
        marker = "  <-- 100"
    elif edges[i] <= 500 < edges[i + 1]:
        marker = "  <-- 500"
    print(f"  [{edges[i]:9.1f},{edges[i+1]:9.1f}) {hist[i]:8d} {bar}{marker}")

print("\n=== does total_input_bytes just track tx_count (frequency), not call complexity? ===")
print(f"Pearson corr(total_input_bytes, tx_count): {tib.corr(all_df['tx_count']):.4f}")
print(f"Pearson corr(total_input_bytes, min_input_bytes): {tib.corr(all_df['min_input_bytes']):.4f}")
per_tx = (tib / all_df["tx_count"]).replace([np.inf, -np.inf], np.nan).dropna()
print("\nper-tx average bytes (total_input_bytes / tx_count) distribution:")
print(per_tx.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== among edges with total_input_bytes in [100,500) (the 'mediumInput' bucket): tx_count breakdown ===")
medium = all_df[(tib >= 100) & (tib < 500)]
print(medium["tx_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
print(f"fraction with tx_count == 1 (genuinely single-call edges): {(medium['tx_count'] == 1).mean():.4f}")
print(f"fraction with tx_count >= 5: {(medium['tx_count'] >= 5).mean():.4f}")

print("\n\nDONE.")
