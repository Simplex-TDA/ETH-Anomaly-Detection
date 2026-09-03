"""Progress tracker for scripts_fetch_all_years.py (the FINAL-corrected
2020-2025 ETH + ERC20 refetch).

Coarse progress: counts distinct completed days per (year, source) by
reading each source's _progress.json (chunk keys are "{date}_{start}_{end}",
so the date prefix gives day-level completion regardless of how many
chunks that day happened to split into).

Usage:
    python3 scripts/monitoring/check_fetch_progress.py
"""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/ (this script lives in scripts/monitoring/)
os.chdir(ROOT)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SOURCES = ["eth_tx_value_output", "erc20_tx_value_output"]
LOG_FILE = Path("logs/fetch_all_years_FINAL.log")

DAYS_IN_YEAR = {y: 366 if y % 4 == 0 else 365 for y in YEARS}


def days_done(year, source):
    f = Path(f"data/{year}/{source}/_progress.json")
    if not f.exists():
        return 0
    try:
        chunks = json.loads(f.read_text())["completed_chunks"]
    except (json.JSONDecodeError, OSError, KeyError):
        return 0
    dates = {c.split("_")[0] for c in chunks}
    return len(dates)


def current_position():
    """Last '== YEAR ==' and '== date ==' lines from the log."""
    if not LOG_FILE.exists():
        return None, None
    text = LOG_FILE.read_text(errors="replace")
    years = re.findall(r"YEAR (\d{4})", text)
    dates = re.findall(r"══ (\d{4}-\d{2}-\d{2}) ══", text)
    stage = "ERC20" if "ERC20: downloading" in text.split("YEAR " + years[-1])[-1] else "ETH"
    return (years[-1] if years else None, dates[-1] if dates else None, stage if years else None)


def main():
    print("=== Fetch progress (FINAL-corrected 2020-2025 refetch) ===\n")
    grand_total = 0
    grand_done = 0
    for year in YEARS:
        total = DAYS_IN_YEAR[year]
        row = []
        for source in SOURCES:
            done = days_done(year, source)
            grand_total += total
            grand_done += done
            row.append(f"{source}: {done}/{total}")
        print(f"  {year}: " + "  |  ".join(row))

    pct = 100 * grand_done / grand_total if grand_total else 0
    print(f"\nOverall: {grand_done}/{grand_total} year-days-source completed ({pct:.1f}%)")

    year, date, stage = current_position()
    if year:
        print(f"\nCurrently on: YEAR {year}, {stage or '?'}, last day seen: {date or '?'}")

    if LOG_FILE.exists():
        age_s = time.time() - LOG_FILE.stat().st_mtime
        print(f"\nLog last updated {age_s:.0f}s ago", end="")
        if age_s > 300:
            print("  [!] no log activity in 5+ min -- check the process is alive")
        else:
            print()


if __name__ == "__main__":
    main()
