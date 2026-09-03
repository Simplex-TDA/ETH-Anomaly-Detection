"""Progress tracker for run_tda_all_years.py.

Coarse progress: counts completed (year, run_name) combos by reading the
run_results_V2_<bucket>_<year>.json files directly (a combo only appears
once its layer/weight/year finishes -- save_run_results is called once
per combo, not per-day). One file per weight-group bucket (tx_count /
tx_value / gasfees), matching run_tda_all_years.py's split.

Fine progress: parses the last 'T-step: building PDs' tqdm line out of
run_tda_all_years.log to show day-level progress within whichever combo
is currently running (tqdm writes \r-separated updates, not newlines).

Usage:
    python3 check_tda_progress.py
"""
import json
import re
import time
from pathlib import Path

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LOG_FILE = Path("run_tda_all_years_2024_2025.log")
RESULTS_PREFIX_BASE = "run_results_V2"

EXPECTED = {
    "tx_count": 7,
    "tx_value": 1,
    "gasfees":  6,
}
TOTAL_EXPECTED = sum(EXPECTED.values()) * len(YEARS)  # 56


def completed_combos():
    done = {}
    for year in YEARS:
        names = []
        for bucket in EXPECTED:
            f = Path(f"{RESULTS_PREFIX_BASE}_{bucket}_{year}.json")
            if f.exists():
                try:
                    names.extend(json.loads(f.read_text()).keys())
                except json.JSONDecodeError:
                    pass
        done[year] = sorted(names)
    return done


def current_day_progress():
    """Last 'T-step: building PDs' tqdm state from the log, if any."""
    if not LOG_FILE.exists():
        return None
    text = LOG_FILE.read_text(errors="replace")
    chunks = text.split("T-step: building PDs")
    if len(chunks) < 2:
        return None
    last = chunks[-1]
    # tqdm writes \r-separated updates within this chunk; take the last one
    last_update = last.split("\r")[-1].split("\n")[0]
    m = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", last_update)
    if not m:
        return None
    pct, cur, total, timing = m.groups()
    return {"pct": int(pct), "cur": int(cur), "total": int(total), "timing": timing}


def current_run_name():
    """Most recent 'Run "<name>"' line in the log."""
    if not LOG_FILE.exists():
        return None
    matches = re.findall(r'Run "([^"]+)"', LOG_FILE.read_text(errors="replace"))
    return matches[-1] if matches else None


def main():
    done = completed_combos()
    n_done = sum(len(v) for v in done.values())

    print(f"=== TDA rebuild progress: {n_done}/{TOTAL_EXPECTED} combos complete "
          f"({100*n_done/TOTAL_EXPECTED:.1f}%) ===\n")

    for year in YEARS:
        n_year_expected = sum(EXPECTED.values())
        print(f"  {year}: {len(done[year])}/{n_year_expected}  -> {done[year]}")

    run_name = current_run_name()
    day_prog = current_day_progress()
    print()
    if run_name:
        print(f"Currently running: {run_name}")
    if day_prog:
        print(f"  Day progress: {day_prog['cur']}/{day_prog['total']} "
              f"({day_prog['pct']}%)  [{day_prog['timing']}]")
    elif run_name:
        print("  (no day-level progress line yet -- likely still loading data)")

    if LOG_FILE.exists():
        age_s = time.time() - LOG_FILE.stat().st_mtime
        print(f"\nLog last updated {age_s:.0f}s ago", end="")
        if age_s > 300:
            print("  [!] no log activity in 5+ min -- check the process is alive")
        else:
            print()


if __name__ == "__main__":
    main()
