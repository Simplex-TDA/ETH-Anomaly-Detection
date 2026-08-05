"""Quick progress check for run_ranking_all_years.py -- run anytime:

    python3 check_progress.py

Computes EXACT completion: for each (filter, ranking_metric), how many
distinct dates are covered vs. days-in-year. A prior version estimated
from row counts (rows-so-far / (top_n * days)), which is WRONG -- many
layers (especially quieter ones like the _ERC20_only filters) have fewer
than top_n=2000 nodes in their top-N-edge graph most days, so they never
produce the full 2000 rows/day even when 100% complete. That version's
progress bar would plateau below 100% forever, never actually reaching it,
even on already-finished years -- confirmed directly: 2020's Mode A/B rows
looked ~96-100% "done" by the row-count estimate, but every single one of
its 116 (filter, ranking_metric) combos already has full 366/366-day
coverage. Date coverage is the metric that's actually correct.

This reads full columns (not just metadata), so it's slower than a pure
metadata check (~15s/year for an already-complete year) -- worth it for
correctness. Handles the run-in-progress race gracefully: the running job
overwrites each file whole on every append, so a file can briefly have no
valid parquet footer if this script's read lands mid-write -- such files
are skipped for this check (noted on stderr) rather than crashing.
"""
import sys
from pathlib import Path

import pandas as pd

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

N_MODE_A_COMBOS = 14 + 3 + 12  # tx_count (all 14) + tx_value (3 simple_txs_*) + total_gas_fees (12 non-ERC20_only)
N_MODE_B_COMBOS = N_MODE_A_COMBOS * 4  # x page_rank/strength/k_core/clustering


def bar(frac: float, width: int = 30) -> str:
    frac = min(1.0, frac)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac*100:5.1f}%"


def read_safe(files):
    frames, skipped = [], []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=["filter", "ranking_metric", "date"]))
        except Exception:
            skipped.append(f.name)
    return frames, skipped


def coverage_frac(files, n_combos_expected: int, n_days: int, label: str, year: int):
    if not files:
        return 0.0, 0, 0
    frames, skipped = read_safe(files)
    if skipped:
        print(f"  [{year}] note: {len(skipped)} {label} file(s) mid-write, skipped this check: {skipped}", file=sys.stderr)
    if not frames:
        return 0.0, 0, 0
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    coverage = df.groupby(["filter", "ranking_metric"])["date"].nunique()
    n_complete = int((coverage >= n_days).sum())
    # overall fraction: total (combo, date) pairs present / total expected
    total_present = int(coverage.clip(upper=n_days).sum())
    frac = min(1.0, total_present / (n_combos_expected * n_days))
    return frac, n_complete, len(coverage)


def year_progress(year: int) -> dict:
    daily_dir = Path(f"data/ranking/{year}/daily")
    n_days = 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
    result = {"year": year, "mode_a_frac": 0.0, "mode_b_frac": 0.0,
              "mode_a_complete": 0, "mode_a_seen": 0, "mode_b_complete": 0, "mode_b_seen": 0}

    if not daily_dir.exists():
        return result

    a_files = sorted(daily_dir.glob("daily_ranking_*.parquet"))
    b_files = sorted(daily_dir.glob("daily_centrality_*.parquet"))

    if a_files:
        result["mode_a_frac"], result["mode_a_complete"], result["mode_a_seen"] = coverage_frac(
            a_files, N_MODE_A_COMBOS, n_days, "Mode A", year)
    if b_files:
        result["mode_b_frac"], result["mode_b_complete"], result["mode_b_seen"] = coverage_frac(
            b_files, N_MODE_B_COMBOS, n_days, "Mode B", year)

    return result


results = [year_progress(y) for y in YEARS]

print(f"{'Year':6s} {'Mode A (edge ranking)':45s} {'Mode B (centrality)':45s}")
for r in results:
    a_note = f"{r['mode_a_complete']}/{N_MODE_A_COMBOS} combos fully done"
    b_note = f"{r['mode_b_complete']}/{N_MODE_B_COMBOS} combos fully done"
    print(f"{r['year']:<6d} {bar(r['mode_a_frac']):40s} {bar(r['mode_b_frac']):40s}")
    print(f"{'':6s} {a_note:40s} {b_note:40s}")

overall_a = sum(r["mode_a_frac"] for r in results) / len(results)
overall_b = sum(r["mode_b_frac"] for r in results) / len(results)
overall = (overall_a + overall_b) / 2  # Mode A and Mode B are roughly equal-weight in wall-clock time

print()
print(f"{'OVERALL':6s} {bar(overall_a):40s} {bar(overall_b):40s}")
print()
print(f"Combined progress: {bar(overall, width=50)}")
