"""Progress tracker for run_tda_band_sweep.py.

Coarse progress: counts completed (year, run_name) combos by reading the
run_results_V2_bandsweep_tx_count_<year>.json files directly (a combo only
appears once its layer/band/year finishes).

Fine progress: parses the last 'T-step: building PDs' tqdm line out of the
log file to show day-level progress within whichever combo is currently
running (tqdm writes \\r-separated updates, not newlines).

Usage:
    python3 scripts/monitoring/check_band_sweep_progress.py [log_file]
    (defaults to logs/run_tda_band_sweep.log)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/monitoring/)
os.chdir(ROOT)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LOG_FILE = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/run_tda_band_sweep.log")
RESULTS_PREFIX = "results/run_results_V2_bandsweep_tx_count"

EXPECTED_RUN_NAMES = [
    f"{layer}_{band}"
    for layer in ("contract_nonFactory_ETH_only", "simple_txs_ETH_only")
    for band in ("band1", "band2", "band3")
]
TOTAL_EXPECTED = len(EXPECTED_RUN_NAMES) * len(YEARS)  # 36


def completed_combos():
    done = {}
    for year in YEARS:
        f = Path(f"{RESULTS_PREFIX}_{year}.json")
        names = []
        if f.exists():
            try:
                names = sorted(json.loads(f.read_text()).keys())
            except json.JSONDecodeError:
                pass
        done[year] = names
    return done


def completed_groups():
    """(layer,band) groups where all 6 years are present."""
    done = completed_combos()
    per_group = {name: sum(name in done[y] for y in YEARS) for name in EXPECTED_RUN_NAMES}
    return per_group


def current_day_progress():
    if not LOG_FILE.exists():
        return None
    text = LOG_FILE.read_text(errors="replace")
    chunks = text.split("T-step: building PDs")
    if len(chunks) < 2:
        return None
    last = chunks[-1]
    last_update = last.split("\r")[-1].split("\n")[0]
    m = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", last_update)
    if not m:
        return None
    pct, cur, total, timing = m.groups()
    return {"pct": int(pct), "cur": int(cur), "total": int(total), "timing": timing}


def current_run_name():
    if not LOG_FILE.exists():
        return None
    matches = re.findall(r'Run "([^"]+)"', LOG_FILE.read_text(errors="replace"))
    return matches[-1] if matches else None


def skip_counts():
    if not LOG_FILE.exists():
        return {}
    text = LOG_FILE.read_text(errors="replace")
    return {
        "timeout": text.count("[SKIP-TIMEOUT]"),
        "died": text.count("[SKIP-DIED]"),
        "error": text.count("[SKIP-ERROR]"),
    }


def main():
    groups = completed_groups()
    n_done_groups = sum(1 for v in groups.values() if v == len(YEARS))
    done = completed_combos()
    n_done_combos = sum(len(v) for v in done.values())

    print(f"=== band sweep progress: {n_done_groups}/{len(EXPECTED_RUN_NAMES)} groups complete, "
          f"{n_done_combos}/{TOTAL_EXPECTED} (layer,band,year) combos ({100*n_done_combos/TOTAL_EXPECTED:.1f}%) ===\n")

    for name, n_years in groups.items():
        marker = "DONE" if n_years == len(YEARS) else f"{n_years}/{len(YEARS)} years"
        print(f"  {name:45s} {marker}")

    run_name = current_run_name()
    day_prog = current_day_progress()
    print()
    if run_name:
        print(f"Currently running: {run_name}")
    if day_prog:
        print(f"  Day progress: {day_prog['cur']}/{day_prog['total']} "
              f"({day_prog['pct']}%)  [{day_prog['timing']}]")
    elif run_name:
        print("  (no day-level progress line yet -- likely still loading data for the year)")

    skips = skip_counts()
    if skips and sum(skips.values()) > 0:
        print(f"\nSkip events so far: {skips}")
    else:
        print("\nNo skip/timeout/error events so far.")

    if LOG_FILE.exists():
        age_s = time.time() - LOG_FILE.stat().st_mtime
        print(f"\nLog last updated {age_s:.0f}s ago", end="")
        if age_s > 900:
            print("  [!] no log activity in 15+ min -- check the process is alive "
                  "(a fresh year's raw-data load can take ~8-10min with no output, "
                  "so don't worry until well past that)")
        else:
            print()

    if n_done_groups == len(EXPECTED_RUN_NAMES):
        print("\n*** ALL GROUPS COMPLETE ***")


if __name__ == "__main__":
    main()
