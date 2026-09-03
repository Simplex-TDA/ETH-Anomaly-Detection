"""Idea 16 diagnostic 3, step 1: full 2020-2025 daily raw scalars for the
"dual-active address" signal -- addresses active in BOTH simple_txs
(tx_value>0) and contract_txs (tx_value==0) on the same day. Currently
these are two disconnected observations in two separate diagrams; the
day-sampled check (audit_simple_vs_contract_overlap.py) found ~46% of
contract-calling addresses also do plain transfers same-day -- large
enough to be worth a real confirmatory test, not just a diagnostic.

Raw (unrestricted) address populations, matching that diagnostic's
convention -- this is a market-activity signal, not scoped to the
daily_top=750 node selection any individual layer's own graph uses.

Processes every weekly file (not a sample) across all 6 years -- a full
pass, no Ripser/curvature, just address-set arithmetic per day.
"""
import glob
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
COLS = ["date", "from_addr", "to_addr", "tx_value"]
OUT = Path("results/exploratory/dual_active_daily.parquet")

t0 = time.time()
day_rows = []
for year in YEARS:
    files = sorted(glob.glob(f"data/{year}/eth_tx_value_output/weekly/eth_tx_*.parquet"))
    for f in files:
        tbl = pq.read_table(f, columns=COLS)
        df = tbl.to_pandas()
        for date, day_df in df.groupby("date", sort=False):
            simple = day_df.loc[day_df["tx_value"] > 0]
            contract = day_df.loc[day_df["tx_value"] == 0]
            simple_nodes = set(simple["from_addr"]) | set(simple["to_addr"])
            contract_nodes = set(contract["from_addr"]) | set(contract["to_addr"])
            inter = simple_nodes & contract_nodes
            union = simple_nodes | contract_nodes
            day_rows.append({
                "date": date,
                "n_simple_nodes": len(simple_nodes),
                "n_contract_nodes": len(contract_nodes),
                "n_dual_active": len(inter),
                "dual_active_jaccard": (len(inter) / len(union)) if union else 0.0,
                "dual_active_frac_of_simple": (len(inter) / len(simple_nodes)) if simple_nodes else 0.0,
                "dual_active_frac_of_contract": (len(inter) / len(contract_nodes)) if contract_nodes else 0.0,
            })
        print(f"[{time.time()-t0:6.0f}s] {f}: {df['date'].nunique()} days processed", flush=True)

result = pd.DataFrame(day_rows).drop_duplicates(subset="date").sort_values("date")
result["date"] = pd.to_datetime(result["date"])
result = result.set_index("date")
result.to_parquet(OUT)
print(f"\n[{time.time()-t0:6.0f}s] Total days: {len(result)}, range {result.index.min()} .. {result.index.max()}", flush=True)
print(result.describe())
print("\n\nDONE.")
