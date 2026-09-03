"""Progress tracker for run_tda_all_layers_FINAL.py (the full V1 8-layer +
V2 14-layer diagram rebuild, FINAL-corrected data, 2020-2025).

Coarse progress: counts completed (year, run_name) combos by reading
run_results_V1_<year>.json and run_results_V2_<bucket>_<year>.json
directly. Fine progress: parses the last 'T-step: building PDs' tqdm line
out of the log for day-level progress within whatever combo is running.

Usage:
    python3 scripts/monitoring/check_tda_final_progress.py
"""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/monitoring/)
os.chdir(ROOT)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LOG_FILE = Path("logs/run_tda_all_layers_FINAL.log")

V1_RUN_NAMES = [
    "contract_txs_ETH_only_750", "simple_txs_ETH_only_750", "contract_txs_ALL_750",
    "simple_txs_ALL_750", "contract_factory_ALL_750", "contract_nonFactory_ALL_750",
    "contract_highInput_ALL_750", "contract_mediumInput_ALL_750",
]
V2_EXPECTED = {"tx_count": 7, "tx_value": 1, "gasfees": 6}

TOTAL_EXPECTED = (len(V1_RUN_NAMES) + sum(V2_EXPECTED.values())) * len(YEARS)  # (8 + 14) * 6 = 132


def completed_combos():
    done = {}
    for year in YEARS:
        names = []
        v1_file = Path(f"results/run_results_V1_{year}.json")
        if v1_file.exists():
            try:
                names.extend(json.loads(v1_file.read_text()).keys())
            except json.JSONDecodeError:
                pass
        for bucket in V2_EXPECTED:
            f = Path(f"results/run_results_V2_{bucket}_{year}.json")
            if f.exists():
                try:
                    names.extend(json.loads(f.read_text()).keys())
                except json.JSONDecodeError:
                    pass
        done[year] = sorted(names)
    return done


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


def main():
    done = completed_combos()
    n_done = sum(len(v) for v in done.values())

    print(f"=== V1+V2 diagram rebuild progress: {n_done}/{TOTAL_EXPECTED} combos complete "
          f"({100*n_done/TOTAL_EXPECTED:.1f}%) ===\n")

    for year in YEARS:
        n_year_expected = len(V1_RUN_NAMES) + sum(V2_EXPECTED.values())
        print(f"  {year}: {len(done[year])}/{n_year_expected}")

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
