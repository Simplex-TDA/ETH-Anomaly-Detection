"""Idea 16 diagnostic 1: does the same address show up active in both
simple_txs_ETH_only (tx_value>0) and contract_txs_ETH_only (tx_value==0)
on the same day? Currently these are two disconnected observations in
two separate diagrams -- if the same address routinely straddles both,
that's real cross-layer structure the topology never sees. Same
day-sampled methodology as audit_byte_cutoffs_overlap.py (1 day per
sampled month, 36 days across 2020-2025), no daily_top restriction here
(raw address populations, matching that script's convention) so this is
purely a data-population check, not yet graph-construction-restricted.
"""
import glob
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = ["01", "03", "05", "07", "09", "11"]
COLS = ["date", "from_addr", "to_addr", "tx_value"]

t0 = time.time()
rows = []
for year in YEARS:
    for month in MONTHS:
        pattern = f"data/{year}/eth_tx_value_output/weekly/eth_tx_*_{year}-{month}_W1.parquet"
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        f = matches[0]
        tbl = pq.read_table(f, columns=COLS)
        df = tbl.to_pandas()
        first_date = sorted(df["date"].unique())[0]
        day_df = df[df["date"] == first_date]

        def node_set(mask):
            sub = day_df[mask]
            return set(sub["from_addr"]) | set(sub["to_addr"])

        simple = node_set(day_df["tx_value"] > 0)
        contract = node_set(day_df["tx_value"] == 0)

        inter = simple & contract
        union = simple | contract
        jaccard = len(inter) / len(union) if union else float("nan")
        rows.append({
            "year": year, "month": month, "date": first_date,
            "n_simple": len(simple), "n_contract": len(contract), "n_dual": len(inter),
            "jaccard": jaccard,
            "simple_frac_also_contract": (len(inter) / len(simple)) if simple else float("nan"),
            "contract_frac_also_simple": (len(inter) / len(contract)) if contract else float("nan"),
        })
        print(f"[{time.time()-t0:5.0f}s] {f} day={first_date}: |simple|={len(simple)} |contract|={len(contract)} "
              f"|dual|={len(inter)} J={jaccard:.4f} simple->also_contract={rows[-1]['simple_frac_also_contract']:.4f} "
              f"contract->also_simple={rows[-1]['contract_frac_also_simple']:.4f}", flush=True)

result = pd.DataFrame(rows)
result.to_parquet("results/exploratory/simple_vs_contract_overlap.parquet")

print("\n=== Summary across all sampled days ===")
print(result[["jaccard", "simple_frac_also_contract", "contract_frac_also_simple"]].describe())
print("\nyear-over-year trend:")
print(result.groupby("year")[["simple_frac_also_contract", "contract_frac_also_simple"]].mean())

print("\n\nDONE.")
