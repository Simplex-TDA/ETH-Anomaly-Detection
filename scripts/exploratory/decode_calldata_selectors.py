"""Idea 12b step 2: decode a sample of contract-call transactions' function
selectors (first 4 bytes of calldata) against the openchain.xyz public
4-byte signature database, to check what kinds of function calls actually
populate each total_input_bytes band (nonFactory <100 / mediumInput
100-500 / highInput >=500).

Raw calldata isn't stored locally (the weekly parquet files only keep the
total_input_bytes aggregate, confirmed via schema inspection) -- this pulls
`input` directly from Xatu ClickHouse's canonical_execution_transaction.
No LIMIT-without-ORDER-BY randomness concerns here: exploratory, not a
statistical estimate, same rigor level as steps 1/3's deterministic day
selection.

Reuses the exact same 2 representative weeks as step 5's curvature audit
(audit_byte_cutoffs_curvature.py) for direct comparability across steps:
2021-07-05..11 and 2024-07-01..07 -- one early-network-state week, one
recent one.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, ".")

import requests
import pandas as pd

import credentials as c
from functions.eth_data_fetcher import ch_query, get_block_range_for_date

WEEKS = [
    ("2021-07-05", "2021-07-11"),
    ("2024-07-01", "2024-07-07"),
]

# (label, lower bound inclusive, upper bound exclusive) on n_input_nonzero_bytes
BANDS = [
    ("nonFactory", 1, 100),
    ("mediumInput", 100, 500),
    ("highInput", 500, None),
]

N_PER_BAND = 1000

t0 = time.time()
rows = []

for week_start, week_end in WEEKS:
    r0 = get_block_range_for_date(week_start, c.CLICKHOUSE_HOST, c.CLICKHOUSE_USER, c.CLICKHOUSE_PASSWORD)
    r1 = get_block_range_for_date(week_end, c.CLICKHOUSE_HOST, c.CLICKHOUSE_USER, c.CLICKHOUSE_PASSWORD)
    block_start, block_end = r0[0], r1[1]
    print(f"[{time.time()-t0:6.0f}s] week {week_start}..{week_end}: blocks {block_start:,}-{block_end:,}", flush=True)

    for band_label, lo, hi in BANDS:
        hi_clause = f"AND t.n_input_nonzero_bytes < {hi}" if hi is not None else ""
        sql = f"""
        SELECT
            t.n_input_nonzero_bytes AS n_bytes,
            substring(t.input, 1, 10) AS selector
        FROM default.canonical_execution_transaction AS t FINAL
        WHERE t.meta_network_name = 'mainnet'
          AND t.block_number >= {block_start}
          AND t.block_number <= {block_end}
          AND t.value = 0
          AND t.n_input_nonzero_bytes >= {lo}
          {hi_clause}
        LIMIT {N_PER_BAND}
        """
        df = ch_query(sql, c.CLICKHOUSE_HOST, c.CLICKHOUSE_USER, c.CLICKHOUSE_PASSWORD)
        df["band"] = band_label
        df["week"] = f"{week_start}..{week_end}"
        rows.append(df)
        print(f"[{time.time()-t0:6.0f}s]   {band_label}: {len(df):,} rows pulled", flush=True)

sample = pd.concat(rows, ignore_index=True)
sample = sample.dropna(subset=["selector"])
sample = sample[sample["selector"].str.len() == 10]  # "0x" + 8 hex chars = 4 bytes
print(f"\n[{time.time()-t0:6.0f}s] total sample: {len(sample):,} transactions, "
      f"{sample['selector'].nunique():,} distinct selectors", flush=True)

# ── decode distinct selectors against openchain.xyz, batched ──
distinct_selectors = sample["selector"].unique().tolist()
selector_to_name = {}
BATCH = 100
for i in range(0, len(distinct_selectors), BATCH):
    batch = distinct_selectors[i:i + BATCH]
    resp = requests.get(
        "https://api.openchain.xyz/signature-database/v1/lookup",
        params={"function": ",".join(batch)},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()["result"]["function"]
    for sel in batch:
        matches = result.get(sel) or result.get(sel.lower())
        if matches:
            selector_to_name[sel] = matches[0]["name"]
        else:
            selector_to_name[sel] = None
    print(f"[{time.time()-t0:6.0f}s] decoded batch {i}-{i+len(batch)} "
          f"({sum(v is not None for v in selector_to_name.values())}/{len(selector_to_name)} resolved so far)", flush=True)

sample["decoded_name"] = sample["selector"].map(selector_to_name)
sample["resolved"] = sample["decoded_name"].notna()

OUT = ROOT / "results" / "exploratory" / "calldata_selector_decode.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)
sample.to_parquet(OUT, index=False)
print(f"\n[{time.time()-t0:6.0f}s] saved {len(sample):,} rows -> {OUT}", flush=True)

# ── summary ──
print(f"\n=== resolution rate by band ===")
print(sample.groupby("band")["resolved"].mean().to_string())

print(f"\n=== top 15 decoded function names per band ===")
for band_label, _, _ in BANDS:
    sub = sample[(sample["band"] == band_label) & sample["resolved"]]
    print(f"\n-- {band_label} (n={len(sample[sample['band']==band_label]):,}, "
          f"resolved={sub.shape[0]:,}) --")
    print(sub["decoded_name"].value_counts().head(15).to_string())

print("\n\nDONE.")
