"""Driver: fetch + weekly-aggregate ETH and ERC20 data for 2020-2025.

Resumable by construction (download_eth_transactions/download_erc20_transfers
track completed chunks in a per-year _progress.json) -- safe to re-run if
interrupted, picks up where it left off. Run from the repo root:

    python3 scripts_fetch_all_years.py

Logs progress to stdout; redirect to a file if running unattended.
"""
import sys
import time
sys.path.insert(0, "functions")

from eth_data_fetcher import download_eth_transactions, aggregate_to_weekly
from erc20_data_fetcher import download_erc20_transfers, aggregate_to_weekly as aggregate_erc20_weekly
import credentials

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
CHUNK_SIZE = 1000

for year in YEARS:
    t0 = time.time()
    print(f"\n{'='*80}\nYEAR {year}\n{'='*80}", flush=True)

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    eth_dir = f"./data/{year}/eth_tx_value_output"
    erc20_dir = f"./data/{year}/erc20_tx_value_output"

    print(f"[{year}] ETH: downloading...", flush=True)
    download_eth_transactions(
        start_date=start_date,
        end_date=end_date,
        output_dir=eth_dir,
        clickhouse_host=credentials.CLICKHOUSE_HOST,
        clickhouse_user=credentials.CLICKHOUSE_USER,
        clickhouse_password=credentials.CLICKHOUSE_PASSWORD,
        chunk_size=CHUNK_SIZE,
    )
    print(f"[{year}] ETH: aggregating to weekly...", flush=True)
    aggregate_to_weekly(chunks_dir=f"{eth_dir}/chunks", weekly_dir=f"{eth_dir}/weekly")

    print(f"[{year}] ERC20: downloading...", flush=True)
    download_erc20_transfers(
        start_date=start_date,
        end_date=end_date,
        output_dir=erc20_dir,
        clickhouse_host=credentials.CLICKHOUSE_HOST,
        clickhouse_user=credentials.CLICKHOUSE_USER,
        clickhouse_password=credentials.CLICKHOUSE_PASSWORD,
        chunk_size=CHUNK_SIZE,
    )
    print(f"[{year}] ERC20: aggregating to weekly...", flush=True)
    aggregate_erc20_weekly(chunks_dir=f"{erc20_dir}/chunks", weekly_dir=f"{erc20_dir}/weekly")

    print(f"[{year}] DONE in {time.time()-t0:.0f}s", flush=True)

print("\n\nAll years complete.", flush=True)
