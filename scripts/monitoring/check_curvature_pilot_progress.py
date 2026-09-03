"""Progress tracker for run_curvature_pilot.py.

Reads curvature_pilot_results.parquet directly (saved incrementally after
every (layer, year) combo) plus the log's progress lines.

Usage:
    python3 scripts/monitoring/check_curvature_pilot_progress.py [log_file]
    (defaults to logs/run_curvature_pilot.log)
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/monitoring/)
os.chdir(ROOT)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LAYERS = ["simple_txs_ETH_only", "contract_nonFactory_ETH_only",
          "contract_mediumInput_ETH_only", "contract_highInput_ETH_only",
          "simple_txs_ETH_only_value"]
LOG_FILE = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/run_curvature_pilot.log")
RESULTS_FILE = Path("results/curvature_pilot_results.parquet")
TOTAL_EXPECTED = len(LAYERS) * len(YEARS)


def main():
    if RESULTS_FILE.exists():
        df = pd.read_parquet(RESULTS_FILE)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        done_pairs = df.groupby(["layer", "year"]).size()
        n_done = len(done_pairs)
        print(f"=== curvature pilot progress: {n_done}/{TOTAL_EXPECTED} (layer, year) combos, "
              f"{len(df)} total day-rows ({100*n_done/TOTAL_EXPECTED:.1f}%) ===\n")
        for layer in LAYERS:
            years_done = sorted(y for (l, y) in done_pairs.index if l == layer)
            print(f"  {layer:32s} years done: {years_done}")
        print()
        print("NaN check per numeric column:")
        print(df.select_dtypes("number").isna().sum().to_string())
    else:
        print("=== curvature pilot progress: 0 rows saved yet ===")

    if LOG_FILE.exists():
        text = LOG_FILE.read_text(errors="replace")
        skip_lines = re.findall(r"\[SKIP\].*", text)
        if skip_lines:
            print(f"\n{len(skip_lines)} skip events:")
            for line in skip_lines[-5:]:
                print(f"  {line}")
        else:
            print("\nNo skip events.")

        age_s = time.time() - LOG_FILE.stat().st_mtime
        print(f"Log last updated {age_s:.0f}s ago", end="")
        if age_s > 900:
            print("  [!] no log activity in 15+ min -- check the process is alive "
                  "(a fresh year's raw-data load can take several minutes, so don't worry until well past that)")
        else:
            print()

        if "\nDONE" in text or text.rstrip().endswith("DONE."):
            print("\n*** DONE ***")


if __name__ == "__main__":
    main()
