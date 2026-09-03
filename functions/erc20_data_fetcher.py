"""
ERC20 Token Transfer Data Fetcher
==================================
Downloads ERC20 token transfers from Xatu ClickHouse database.

Downloads in block-range chunks to avoid timeouts.
Each chunk is saved as its own parquet file immediately.
Re-running resumes from where it left off.

Output columns:
  date, start_block, end_block, from_addr, to_addr, erc20, tx_count, tx_value, max_value
"""

import os
import json
import requests
import pandas as pd
from io import StringIO
from tqdm.auto import tqdm
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN DECIMALS
# ═══════════════════════════════════════════════════════════════════════════
# canonical_execution_erc20_transfers has no decimals column (confirmed via
# schema inspection), so per-token scaling has to come from an external,
# hardcoded source. Covers the majors where getting it wrong matters most by
# transfer volume; everything else defaults to 18, the ERC20 standard's
# overwhelming convention. Addresses lowercased to match ClickHouse's storage
# format regardless of any checksumming on the query side.
KNOWN_TOKEN_DECIMALS = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
}


def token_decimals(erc20_addr: str) -> int:
    return KNOWN_TOKEN_DECIMALS.get(str(erc20_addr).lower(), 18)


# ═══════════════════════════════════════════════════════════════════════════
# CLICKHOUSE HTTP HELPER
# ═══════════════════════════════════════════════════════════════════════════

def ch_query(sql: str, host: str, user: str, password: str, timeout: int = 120) -> pd.DataFrame:
    """Execute a ClickHouse query and return results as DataFrame."""
    resp = requests.post(
        host,
        params={"query": sql + " FORMAT TSVWithNames"},
        auth=(user, password),
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ClickHouse error {resp.status_code}:\n{resp.text[:1000]}"
        )
    return pd.read_csv(StringIO(resp.text), sep="\t")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK RANGE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

def get_block_range_for_date(date_str: str, host: str, user: str, password: str) -> tuple[int, int] | None:
    """Return (min_block, max_block) for a given UTC date on mainnet.

    canonical_execution_block is a ReplicatedReplacingMergeTree -- FINAL is
    required or this can see stale, un-merged duplicate rows depending on
    background merge timing.
    """
    sql = f"""
    SELECT
        min(block_number) AS start_block,
        max(block_number) AS end_block
    FROM default.canonical_execution_block FINAL
    WHERE
        meta_network_name = 'mainnet'
        AND toDate(block_date_time) = '{date_str}'
    """
    df = ch_query(sql, host, user, password)
    if df.empty or pd.isna(df["start_block"].iloc[0]):
        return None
    return int(df["start_block"].iloc[0]), int(df["end_block"].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# CHUNK QUERY
# ═══════════════════════════════════════════════════════════════════════════

def query_chunk(block_start: int, block_end: int, date_str: str,
                host: str, user: str, password: str) -> pd.DataFrame:
    """
    Aggregate ERC20 transfers for blocks [block_start, block_end].

    canonical_execution_erc20_transfers is also a ReplicatedReplacingMergeTree
    -- FINAL required for the same reason as the block/transaction tables.
    """
    sql = f"""
    SELECT
        '{date_str}'            AS date,
        min(block_number)       AS start_block,
        max(block_number)       AS end_block,
        from_address            AS from_addr,
        to_address              AS to_addr,
        erc20                   AS erc20,
        count()                 AS tx_count,
        SUM(value)              AS tx_value,
        MAX(value)              AS max_value

    FROM default.canonical_execution_erc20_transfers FINAL
    WHERE
        meta_network_name = 'mainnet'
        AND block_number >= {block_start}
        AND block_number <= {block_end}
    GROUP BY
        from_address,
        to_address,
        erc20
    ORDER BY
        tx_count DESC
    """
    df = ch_query(sql, host, user, password)
    if df.empty:
        return df
    df["start_block"] = df["start_block"].astype("int64")
    df["end_block"]   = df["end_block"].astype("int64")
    df["tx_count"]    = df["tx_count"].astype("int64")

    # Convert to token units, per-token decimals (not a blanket 1e18 --
    # USDT/USDC use 6, WBTC uses 8; see KNOWN_TOKEN_DECIMALS above)
    divisor = df["erc20"].map(lambda a: 10.0 ** token_decimals(a))
    df['tx_value'] = df['tx_value'].astype(float) / divisor
    df['max_value'] = df['max_value'].astype(float) / divisor

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def load_progress(progress_file: str) -> dict:
    """Load progress from JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_chunks": []}


def save_progress(progress: dict, progress_file: str):
    """Save progress to JSON file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def chunk_key(date_str: str, block_start: int, block_end: int) -> str:
    """Generate unique key for a chunk."""
    return f"{date_str}_{block_start}_{block_end}"


def chunk_file(date_str: str, block_start: int, block_end: int, chunks_dir: str) -> str:
    """Generate file path for a chunk."""
    return os.path.join(chunks_dir, f"{chunk_key(date_str, block_start, block_end)}.parquet")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def download_erc20_transfers(
    start_date: str,
    end_date: str,
    output_dir: str,
    clickhouse_host: str,
    clickhouse_user: str,
    clickhouse_password: str,
    chunk_size: int = 1000,
):
    """
    Download ERC20 transfers from ClickHouse for the specified date range.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format (inclusive)
    end_date : str
        End date in YYYY-MM-DD format (inclusive)
    output_dir : str
        Directory to save output parquet files
    clickhouse_host : str
        ClickHouse server URL
    clickhouse_user : str
        ClickHouse username
    clickhouse_password : str
        ClickHouse password
    chunk_size : int
        Number of blocks per query (default: 1000)
    """
    chunks_dir = os.path.join(output_dir, "chunks")
    progress_file = os.path.join(output_dir, "_progress.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    progress = load_progress(progress_file)
    completed_chunks = set(progress["completed_chunks"])

    date_range = pd.date_range(start_date, end_date, freq="D")
    all_dates = [d.strftime("%Y-%m-%d") for d in date_range]

    for date_str in all_dates:
        print(f"\n══ {date_str} ══ looking up block range …", end=" ", flush=True)

        block_range = get_block_range_for_date(date_str, clickhouse_host,
                                                clickhouse_user, clickhouse_password)
        if block_range is None:
            print("no blocks found, skipping.")
            continue

        day_start, day_end = block_range
        total_blocks = day_end - day_start + 1

        # Build list of chunk boundaries for this day
        chunks = []
        b = day_start
        while b <= day_end:
            chunks.append((b, min(b + chunk_size - 1, day_end)))
            b += chunk_size

        todo_chunks = [
            (s, e) for s, e in chunks
            if chunk_key(date_str, s, e) not in completed_chunks
        ]

        print(f"blocks {day_start:,}–{day_end:,} ({total_blocks:,} blocks, "
              f"{len(chunks)} chunks, {len(todo_chunks)} remaining)")

        for block_start, block_end in tqdm(todo_chunks, desc=f"  {date_str}", unit="chunk"):
            key = chunk_key(date_str, block_start, block_end)
            path = chunk_file(date_str, block_start, block_end, chunks_dir)

            try:
                df = query_chunk(block_start, block_end, date_str,
                                clickhouse_host, clickhouse_user, clickhouse_password)
            except Exception as e:
                print(f"\n  ✗  chunk {block_start}–{block_end} failed: {e}")
                continue

            df.to_parquet(path, index=False)
            completed_chunks.add(key)
            progress["completed_chunks"] = list(completed_chunks)
            save_progress(progress, progress_file)

    print(f"\n✅ Download complete → {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# WEEKLY AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_to_weekly(chunks_dir: str, weekly_dir: str):
    """
    Aggregate chunk parquet files into weekly parquet files.

    Parameters
    ----------
    chunks_dir : str
        Directory containing chunk parquet files
    weekly_dir : str
        Directory to save weekly aggregated files
    """
    os.makedirs(weekly_dir, exist_ok=True)

    chunk_files = sorted(Path(chunks_dir).glob("*.parquet"))
    print(f"Found {len(chunk_files):,} chunk files\n")

    # Extract dates from filenames
    records = []
    for f in chunk_files:
        date_str = f.stem.split("_")[0]
        records.append({"path": str(f), "date": pd.Timestamp(date_str)})

    file_df = pd.DataFrame(records)

    # Assign ISO week number and week-of-month
    file_df["iso_week"] = file_df["date"].dt.isocalendar().week.astype(int)
    file_df["iso_year"] = file_df["date"].dt.isocalendar().year.astype(int)
    file_df["year_month"] = file_df["date"].dt.strftime("%Y-%m")
    file_df["week_of_month"] = ((file_df["date"].dt.day - 1) // 7) + 1

    # Group key: iso_year + iso_week
    file_df["week_key"] = (
        file_df["iso_year"].astype(str) + "_W" +
        file_df["iso_week"].astype(str).str.zfill(2)
    )

    week_groups = (
        file_df.groupby("week_key")
        .agg(
            paths=("path", list),
            first_date=("date", "min"),
            iso_week=("iso_week", "first"),
        )
        .reset_index()
    )
    week_groups["year_month"] = week_groups["first_date"].dt.strftime("%Y-%m")
    week_groups["week_of_month"] = ((week_groups["first_date"].dt.day - 1) // 7) + 1
    week_groups["out_filename"] = (
        "eth_tx_" +
        week_groups["iso_week"].astype(str).str.zfill(2) + "_" +
        week_groups["year_month"] + "_W" +
        week_groups["week_of_month"].astype(str) +
        ".parquet"
    )

    print(f"Weeks to aggregate: {len(week_groups)}\n")

    for _, row in tqdm(week_groups.iterrows(), total=len(week_groups),
                       desc="Weeks", unit="week"):

        out_path = os.path.join(weekly_dir, row["out_filename"])

        if os.path.exists(out_path):
            tqdm.write(f"  ✓ (already exists) {row['out_filename']}")
            continue

        tqdm.write(f"  ── {row['out_filename']}  ({len(row['paths'])} chunks) …")

        # Load all chunks for this week
        dfs = []
        for path in row["paths"]:
            dfs.append(pd.read_parquet(path))

        raw = pd.concat(dfs, ignore_index=True)

        # Re-aggregate (note: includes erc20 in groupby)
        weekly = (
            raw
            .groupby(["date", "erc20", "from_addr", "to_addr"], as_index=False)
            .agg(
                start_block=("start_block", "min"),
                end_block=("end_block", "max"),
                tx_count=("tx_count", "sum"),
                tx_value=("tx_value", "sum"),
                max_value=("max_value", "max")
            )
            [["date", "start_block", "end_block", "from_addr", "to_addr", "erc20",
              "tx_count", "tx_value", "max_value"]]
            .sort_values(["date", "tx_count"], ascending=[True, False])
        )

        weekly.to_parquet(out_path, index=False)
        tqdm.write(f"     → {len(weekly):,} rows saved to {row['out_filename']}")

    weekly_files = sorted(Path(weekly_dir).glob("*.parquet"))
    print(f"\n✅ Done — {len(weekly_files)} weekly files in {weekly_dir}/")
    for f in weekly_files:
        df = pd.read_parquet(f)
        print(f"   {f.name:55s}  {len(df):>10,} rows")


print('erc20_data_fetcher.py loaded ✓')
