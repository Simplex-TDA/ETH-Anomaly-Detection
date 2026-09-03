"""Idea 12b, step 3: does the nonFactory/mediumInput/highInput 3-way
byte-cutoff split actually produce distinct address populations, or do
the same addresses just get shuffled between buckets day to day? Same
Jaccard-overlap methodology already used for the tx_count-vs-tx_value
ranking-divergence check (found ~2-5% overlap there -- genuinely disjoint
populations, which is why that split earned separate treatment).

One representative day per sampled month (same 36 weekly files as
audit_byte_cutoffs.py, first date present in each), node = either
endpoint (from_addr or to_addr) of a contract-call edge, per the three
current layer filters exactly as defined in run_tda_all_layers_FINAL.py:
  nonFactory:   tx_value==0 & total_input_bytes <  100
  mediumInput:  tx_value==0 & 100 <= total_input_bytes < 500
  highInput:    tx_value==0 & total_input_bytes >= 500
"""
import glob
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = ["01", "03", "05", "07", "09", "11"]
COLS = ["date", "from_addr", "to_addr", "tx_value", "total_input_bytes"]

t0 = time.time()
day_rows = []
for year in YEARS:
    for month in MONTHS:
        pattern = f"data/{year}/eth_tx_value_output/weekly/eth_tx_*_{year}-{month}_W1.parquet"
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        f = matches[0]
        tbl = pq.read_table(f, columns=COLS)
        df = tbl.to_pandas()
        df = df[df["tx_value"] == 0]
        first_date = sorted(df["date"].unique())[0]
        day_df = df[df["date"] == first_date]

        def node_set(mask):
            sub = day_df[mask]
            return set(sub["from_addr"]) | set(sub["to_addr"])

        nf = node_set(day_df["total_input_bytes"] < 100)
        mi = node_set((day_df["total_input_bytes"] >= 100) & (day_df["total_input_bytes"] < 500))
        hi = node_set(day_df["total_input_bytes"] >= 500)

        def jaccard(a, b):
            if not a and not b:
                return float("nan")
            return len(a & b) / len(a | b)

        day_rows.append({
            "year": year, "month": month, "date": first_date,
            "n_nonFactory": len(nf), "n_mediumInput": len(mi), "n_highInput": len(hi),
            "jaccard_nf_mi": jaccard(nf, mi), "jaccard_nf_hi": jaccard(nf, hi), "jaccard_mi_hi": jaccard(mi, hi),
            "mi_frac_also_in_nf": (len(mi & nf) / len(mi)) if mi else float("nan"),
            "hi_frac_also_in_nf": (len(hi & nf) / len(hi)) if hi else float("nan"),
            "hi_frac_also_in_mi": (len(hi & mi) / len(hi)) if hi else float("nan"),
        })
        print(f"[{time.time()-t0:5.0f}s] {f} day={first_date}: "
              f"|nF|={len(nf)} |mI|={len(mi)} |hI|={len(hi)} "
              f"J(nF,mI)={day_rows[-1]['jaccard_nf_mi']:.4f} J(nF,hI)={day_rows[-1]['jaccard_nf_hi']:.4f} "
              f"J(mI,hI)={day_rows[-1]['jaccard_mi_hi']:.4f}", flush=True)

result = pd.DataFrame(day_rows)
result.to_parquet("results/exploratory/byte_cutoff_overlap.parquet")

print("\n=== Summary across all sampled days ===")
print(result[["jaccard_nf_mi", "jaccard_nf_hi", "jaccard_mi_hi",
              "mi_frac_also_in_nf", "hi_frac_also_in_nf", "hi_frac_also_in_mi"]].describe())

print("\n\nDONE.")
