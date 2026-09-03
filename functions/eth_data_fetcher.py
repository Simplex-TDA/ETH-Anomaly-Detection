"""
ETH Transaction Data Fetcher
=============================
Downloads ETH native transactions from Xatu ClickHouse database.

Downloads in block-range chunks to avoid timeouts.
Each chunk is saved as its own parquet file immediately.
Re-running resumes from where it left off.

Output columns:
  date, start_block, end_block, from_addr, to_addr, tx_count, tx_value,
  max_value, total_gas_price, total_gas_fees, total_input_bytes,
  min_input_bytes, factory_approx

total_gas_fees is the actual ETH burned on gas (gas_used * effective price
paid per unit gas), not to be confused with total_gas_price (sum of the
raw gas_price bid field, confounded by tx_count and not the same as what
was actually paid post-EIP-1559). effective price paid = gas_price for
legacy (type 0/1) transactions; for type-2 (EIP-1559, live since the
London hard fork, ~2021-08-05) it's
min(max_fee_per_gas, base_fee_per_gas + max_priority_fee_per_gas), which
needs base_fee_per_gas from canonical_execution_block (joined per block,
scoped to the same block-range chunk already being queried).
"""

import os
import json
import requests
import pandas as pd
from io import StringIO
from tqdm.auto import tqdm
from pathlib import Path


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
    background merge timing (confirmed empirically: the same query without
    FINAL returned a materially different row count than with it, for
    identical historical, already-finalized blocks).
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
    Aggregate transactions for blocks [block_start, block_end].
    Uses a subquery on the block table to avoid a distributed JOIN for the
    day-boundary lookup; the base_fee_per_gas join below is scoped to the
    same small block-range chunk (~1000 blocks) already being queried, so
    it stays a small, local join rather than a distributed one.

    Both tables are ReplicatedReplacingMergeTree -- FINAL required on both
    (verified: without FINAL, the same block range returns duplicate-
    inflated rows depending on background merge state; ClickHouse syntax
    requires FINAL to come after the alias -- `AS t FINAL`, not `FINAL AS t`).
    """
    sql = f"""
    SELECT
        '{date_str}'            AS date,
        min(t.block_number)     AS start_block,
        max(t.block_number)     AS end_block,
        t.from_address          AS from_addr,
        t.to_address            AS to_addr,
        count()                 AS tx_count,
        SUM(t.value)             AS tx_value,
        MAX(t.value)             AS max_value,
        SUM(t.gas_price)         AS total_gas_price,
        SUM(t.gas_used * if(
            t.transaction_type = 2,
            least(t.max_fee_per_gas, coalesce(b.base_fee_per_gas, 0) + t.max_priority_fee_per_gas),
            t.gas_price
        ))                       AS total_gas_fees,
        SUM(t.n_input_nonzero_bytes) AS total_input_bytes,
        MIN(t.n_input_nonzero_bytes) AS min_input_bytes,
        SUM(multiIf(t.value = 0 AND t.n_input_nonzero_bytes > 200, 1, 0)) AS factory_approx

    FROM default.canonical_execution_transaction AS t FINAL
    GLOBAL LEFT JOIN (
        SELECT block_number, base_fee_per_gas
        FROM default.canonical_execution_block FINAL
        WHERE meta_network_name = 'mainnet'
          AND block_number >= {block_start}
          AND block_number <= {block_end}
    ) AS b
    ON t.block_number = b.block_number
    WHERE
        t.meta_network_name = 'mainnet'
        AND t.block_number >= {block_start}
        AND t.block_number <= {block_end}
    GROUP BY
        t.from_address,
        t.to_address
    ORDER BY
        tx_count DESC
    """
    df = ch_query(sql, host, user, password)
    if df.empty:
        return df
    df["start_block"] = df["start_block"].astype("int64")
    df["end_block"]   = df["end_block"].astype("int64")
    df["tx_count"]    = df["tx_count"].astype("int64")

    # Convert wei to ETH
    df['tx_value'] = df['tx_value'].astype(float) / 1e18
    df['max_value'] = df['max_value'].astype(float) / 1e18
    df['total_gas_fees'] = df['total_gas_fees'].astype(float) / 1e18

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

def download_eth_transactions(
    start_date: str,
    end_date: str,
    output_dir: str,
    clickhouse_host: str,
    clickhouse_user: str,
    clickhouse_password: str,
    chunk_size: int = 1000,
):
    """
    Download ETH transactions from ClickHouse for the specified date range.

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

        # Re-aggregate
        weekly = (
            raw
            .groupby(["date", "from_addr", "to_addr"], as_index=False)
            .agg(
                start_block=("start_block", "min"),
                end_block=("end_block", "max"),
                tx_count=("tx_count", "sum"),
                tx_value=("tx_value", "sum"),
                max_value=("max_value", "max"),
                total_gas_price=("total_gas_price", "sum"),
                total_gas_fees=("total_gas_fees", "sum"),
                total_input_bytes=("total_input_bytes", "sum"),
                min_input_bytes=("min_input_bytes", "min"),
                factory_approx=("factory_approx", "sum")
            )
            [["date", "start_block", "end_block", "from_addr", "to_addr", "tx_count",
              "tx_value", "max_value", "total_gas_price", "total_gas_fees",
              "total_input_bytes", "min_input_bytes", "factory_approx"]]
            .sort_values(["date", "tx_count"], ascending=[True, False])
        )

        weekly.to_parquet(out_path, index=False)
        tqdm.write(f"     → {len(weekly):,} rows saved to {row['out_filename']}")

    weekly_files = sorted(Path(weekly_dir).glob("*.parquet"))
    print(f"\n✅ Done — {len(weekly_files)} weekly files in {weekly_dir}/")
    for f in weekly_files:
        df = pd.read_parquet(f)
        print(f"   {f.name:55s}  {len(df):>10,} rows")


print('eth_data_fetcher.py loaded ✓')
