"""Follow-up to the curvature pilot: contract_mediumInput_ETH_only showed
the weakest curvature signal of any layer (max |r| with vol_regime_h7 was
0.058, vs 0.11-0.18 for every other layer), and its daily graphs are the
sparsest tested (~144 edges/day at daily_top=750 after the correct
per-layer filter). Its raw edge count also scales *faster* with daily_top
than any other layer (4.8x from 500->1250, vs 3.4x for nonFactory) --
meaning it isn't structurally capped, just under-fed at 750. This script
tests whether a denser daily graph (higher daily_top) gives curvature a
more stable signal for this specific layer, at daily_top = 500, 1000,
1250 (750 is already computed, in curvature_pilot_results.parquet).

Same construction as run_curvature_pilot.py (correct layer-filter
application, same norm_similarity weight transform, same max_nodes=1000
cap, same fork-based per-day timeout), just varying daily_top instead of
holding it fixed, and scoped to one layer only.

    python3 scripts/runs/run_curvature_mediumInput_dailytop.py
"""
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent           # scripts/runs/ (holds this file and run_curvature_pilot.py)
ROOT = SCRIPT_DIR.parents[1]                            # ETH Anomaly Detection/
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(ROOT)
sys.path.insert(0, "functions")

import pandas as pd
from tad_ethereum_functions import load_filtered_graph

from run_curvature_pilot import curvature_for_day  # reuses the exact same curvature computation, sibling file in scripts/runs/

FILTER_FN = lambda d: (d["tx_value"] == 0) & (d["total_input_bytes"] >= 100) & (d["total_input_bytes"] < 500) & (d["erc20"] == "ETH")
BASE_FILTER_NAME = "contract_mediumInput_ETH_only"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DAILY_TOP_VALUES = [500, 1000, 1250]  # 750 already computed in curvature_pilot_results.parquet
OUT = Path("results/curvature_mediumInput_dailytop_results.parquet")


def existing_results():
    if OUT.exists():
        return pd.read_parquet(OUT)
    return pd.DataFrame(columns=['layer', 'date'])


def main():
    t0 = time.time()
    done = existing_results()
    all_rows = [done] if not done.empty else []

    for daily_top in DAILY_TOP_VALUES:
        layer_key = f"{BASE_FILTER_NAME}_{daily_top}"
        for year in YEARS:
            done_dates = set(done[(done['layer'] == layer_key)]['date']) if not done.empty else set()

            ranking_dir = Path("data/ranking") / str(year)
            filtered_df = load_filtered_graph(
                eth_dir=Path(f"data/{year}/eth_tx_value_output/weekly"),
                erc20_dir=Path(f"data/{year}/erc20_tx_value_output/weekly"),
                global_file=ranking_dir / 'global_top_nodes.parquet',
                daily_dir=ranking_dir / 'daily',
                filters=[BASE_FILTER_NAME],
                ranking_metric='tx_count',
                global_top_num=0,
                daily_top_num=daily_top,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
            )
            if filtered_df.empty:
                print(f"[{time.time()-t0:6.0f}s] daily_top={daily_top} {year}: no data", flush=True)
                continue
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            dates = sorted(filtered_df['date'].unique())

            rows = []
            for d in dates:
                date_str = pd.Timestamp(d).date().isoformat()
                if date_str in done_dates:
                    continue
                day_df = filtered_df[filtered_df['date'] == d]
                day_df = day_df[FILTER_FN(day_df)]
                if len(day_df) < 3:
                    continue
                result, err = curvature_for_day(day_df, edge_weight_col='tx_count')
                if err is not None:
                    print(f"  [SKIP] daily_top={daily_top} {date_str}: {err}", flush=True)
                    continue
                if result is None:
                    continue
                result['layer'] = layer_key
                result['daily_top'] = daily_top
                result['date'] = date_str
                rows.append(result)

            if rows:
                all_rows.append(pd.DataFrame(rows))
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_parquet(OUT)
                all_rows = [combined]
            print(f"[{time.time()-t0:6.0f}s] daily_top={daily_top} {year}: {len(rows)} new days computed "
                  f"({len(dates)} total dates in filtered data)", flush=True)

    print(f"\n[{time.time()-t0:6.0f}s] DONE.", flush=True)


if __name__ == "__main__":
    main()
