"""DuckDB rewrite of run_trailing_rank_pilot.py's ranking step -- after six
Python-level memory fixes each hit the same wall around the 2024 activity
surge, the real problem was never a fixable Python inefficiency: it's
genuinely large data (505.8M raw edge-day rows for this one layer across 6
years) that needs an out-of-core engine, not more dict-tuning in a single
process.

Same semantics as the Python version: rank (from_addr, to_addr) pairs by
trailing-7-calendar-day summed tx_count (a range join, not a plain window
function -- see the comment below for why that distinction is load-bearing),
take the daily top-750 by that trailing sum, then keep each day's OWN edges
(not trailing-aggregated) where both endpoints are in that day's qualifying
set.

v2: per-year resumable (each year's result written once the whole year
completes, skip-if-exists per year).

v3: per-CHUNK resumable. v2's per-year granularity meant a kill anywhere
inside the trailing_ranked/top_pairs/qualifying_nodes/COPY stage for a
year lost 100% of that year's progress, however far in it got -- confirmed
expensively on year 2025 (104M raw rows, this pipeline's heaviest year):
a 44-minute run reached deep into the chunk loop and was killed by the
memory watchdog one step before the year-level save, losing the entire
44 minutes. The actual per-date computation was already chunk-independent
(each date's final qualifying-edge selection only needs that date's own
trailing_ranked/top_pairs/qualifying_nodes rows, built from a JOIN whose
right-hand side is explicitly date-filtered) -- v2 just didn't take
advantage of that, building one full-year trailing_ranked/top_pairs/
qualifying_nodes table before writing anything. v3 does the full
trailing_ranked -> top_pairs -> qualifying_nodes -> qualifying-edges
pipeline PER CHUNK and writes each chunk's result to its own small
parquet file immediately, skipping any chunk whose file already exists.
raw_edges (the month-chunked load) still has to be redone from scratch on
a restart mid-year -- that's a real but much smaller cost (a few minutes,
not tens) than losing the whole chunk loop. Once every chunk for a year
is present, they're concatenated into that year's single edges_{year}.parquet
(the same output shape v1/v2 produced) and the chunk files are removed.
Bonus: since each chunk's trailing_ranked/top_pairs/qualifying_nodes
tables only ever hold ONE chunk's worth of dates instead of the full
year's, peak memory during this stage should also be lower, not just more
resumable.

Set VALIDATE=True first to check DuckDB's output against the Python
reference's known-correct cumulative count (33,263 qualifying rows after
the first 50 days) before trusting a change at full scale.
"""
import shutil
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]  # ETH Anomaly Detection/
OUT_DIR = ROOT / "results" / "exploratory" / "trailing_rank_by_year"
DB_FILE = "/tmp/duckdb_spill/trailing_rank.duckdb"

VALIDATE = False  # re-validated final overnight config (2GB budget, 1-day chunks, per-chunk
# resumable v3): still 33,251
VALIDATE_END_DATE = "2020-02-19"  # the 50th day in the Python reference run

WINDOW_DAYS = 7
TOP_N_EDGES = 750
MEMORY_LIMIT = "2GB"  # lowered from 3GB for year 2025 specifically -- this is the heaviest
# year (104M raw rows, the same activity-surge pattern documented elsewhere in this project)
# and kept dying even during the early month-chunked raw_edges load, well before the
# trailing_ranked stage 3GB was originally tuned for, across 25+ consecutive watchdog-triggered
# retries. Repeated failures at inconsistent points suggest volatile system-wide pressure, not
# a deterministic query-size problem -- a smaller ask improves the odds of fitting whatever's
# actually available at any given moment; the self-healing watchdog handles the rest via
# retries while unsupervised overnight, and v3's per-chunk resumability (see module docstring)
# means a kill no longer costs anywhere near as much when it does happen.
CHUNK_DAYS = 1  # each chunk = its own resumable unit (own output file); also each chunk's
# trailing_ranked/top_pairs/qualifying_nodes tables now only hold this many dates at a time
# (not the full year), which should lower peak memory during this stage too
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()


def load_year_raw_edges(con, year, table_name, month_cache_dir):
    """v3.1: month-level cache, resumable. Every version through v3 rebuilt
    raw_edges from scratch (re-running the SUM/GROUP BY aggregation over
    every month's raw source files) on EVERY restart within a year -- and
    confirmed live overnight to be exactly where year 2025 (104M raw rows)
    now consistently dies: 6+ consecutive kills, zero chunks ever reached,
    always during this load, never even printing 'raw_edges loaded'. Fixed
    the same way the ranking stage was fixed: cache each month's already-
    aggregated result to its own small parquet file the first time it's
    computed, skip-if-exists on every subsequent attempt. raw_edges becomes
    a lazy VIEW over the cached month files (a cheap read+union) instead of
    a persisted TABLE rebuilt via GROUP BY every single time. A kill now
    only costs the ONE month currently being aggregated, not the whole
    year's worth of months already cached from earlier attempts."""
    month_cache_dir.mkdir(parents=True, exist_ok=True)
    date_filter = f"AND date::DATE <= DATE '{VALIDATE_END_DATE}'" if VALIDATE else ""

    chunks = [(year - 1, 12)] + [(year, m) for m in range(1, 13)]  # always try prior December
    # for the lookback, glob just no-ops if absent
    for y, m in chunks:
        if VALIDATE and (y, m) not in [(year - 1, 12), (year, 1), (year, 2), (year, 3)]:
            continue
        cache_file = month_cache_dir / f"{y}-{m:02d}.parquet"
        if cache_file.exists():
            continue  # already aggregated in a prior attempt, reused as-is below
        month_glob = str(ROOT / "data" / str(y) / "eth_tx_value_output" / "weekly" / f"eth_tx_*_{y}-{m:02d}_*.parquet")
        if not list(Path(ROOT / "data" / str(y) / "eth_tx_value_output" / "weekly").glob(f"eth_tx_*_{y}-{m:02d}_*.parquet")):
            continue
        # Write to a .tmp path and rename (atomic on POSIX) only once COPY
        # finishes -- a bare COPY straight to cache_file left a truncated,
        # unreadable parquet file behind when SIGKILL landed mid-write,
        # which the exists() check above then wrongly treated as "already
        # done" on the next attempt (crashed the whole pipeline reading it,
        # not just cost that one month) -- confirmed live on 2025-01.parquet.
        tmp_file = cache_file.with_suffix(".tmp")
        con.execute(f"""
        COPY (
            SELECT date::DATE AS date, from_addr, to_addr, SUM(tx_count) AS tx_count
            FROM read_parquet('{month_glob}')
            WHERE tx_value = 0 AND total_input_bytes < 100 {date_filter}
              AND date::DATE >= DATE '{year}-01-01' - INTERVAL '{WINDOW_DAYS - 1}' DAY
            GROUP BY date::DATE, from_addr, to_addr
        ) TO '{tmp_file}' (FORMAT PARQUET)
        """)
        tmp_file.rename(cache_file)

    cache_glob = str(month_cache_dir / "*.parquet")
    con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{cache_glob}')")


def process_chunk(con, chunk_start, chunk_end, chunk_out):
    """Full trailing_ranked -> top_pairs -> qualifying_nodes -> qualifying-edges
    pipeline for ONE chunk's date range only, writing straight to chunk_out.
    Range join, not a plain window function: a window function only produces
    a trailing sum on dates where that exact pair already has a row, silently
    excluding a pair from a day's ranking if it wasn't active that specific
    day even though it was active within the trailing window -- a real bug
    caught by the very first validation run, not a tie-break difference.
    raw_edges is explicitly date-filtered on BOTH sides of the join (not just
    relying on the ON clause) -- DuckDB doesn't reliably prune an unfiltered
    side even when the join predicate logically bounds it. Writes to a .tmp
    path and renames (atomic on POSIX) only once COPY finishes, same reason
    as load_year_raw_edges's month cache: a SIGKILL mid-write must not leave
    a truncated file at chunk_out, or the caller's skip-if-exists check
    would wrongly treat a corrupt chunk as done on the next attempt."""
    con.execute("CREATE OR REPLACE TEMP TABLE trailing_ranked (date DATE, from_addr VARCHAR, to_addr VARCHAR, trailing_sum BIGINT)")
    con.execute(f"""
    INSERT INTO trailing_ranked
    SELECT c.date, r.from_addr, r.to_addr, SUM(r.tx_count) AS trailing_sum
    FROM (SELECT DISTINCT date FROM raw_edges
          WHERE date BETWEEN DATE '{chunk_start}' AND DATE '{chunk_end}') c
    JOIN (SELECT * FROM raw_edges
          WHERE date BETWEEN DATE '{chunk_start}' - INTERVAL '{WINDOW_DAYS - 1}' DAY AND DATE '{chunk_end}') r
      ON r.date BETWEEN c.date - INTERVAL '{WINDOW_DAYS - 1}' DAY AND c.date
    GROUP BY c.date, r.from_addr, r.to_addr
    """)

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE top_pairs AS
    SELECT date, from_addr, to_addr FROM (
        SELECT date, from_addr, to_addr,
               ROW_NUMBER() OVER (PARTITION BY date ORDER BY trailing_sum DESC, from_addr, to_addr) AS rnk
        FROM trailing_ranked
    )
    WHERE rnk <= {TOP_N_EDGES}
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE qualifying_nodes AS
    SELECT DISTINCT date, addr FROM (
        SELECT date, from_addr AS addr FROM top_pairs
        UNION ALL
        SELECT date, to_addr AS addr FROM top_pairs
    )
    """)

    tmp_out = chunk_out.with_suffix(".tmp")
    con.execute(f"""
    COPY (
        SELECT r.date, r.from_addr, r.to_addr, r.tx_count
        FROM raw_edges r
        JOIN qualifying_nodes qf ON r.date = qf.date AND r.from_addr = qf.addr
        JOIN qualifying_nodes qt ON r.date = qt.date AND r.to_addr = qt.addr
        WHERE r.date BETWEEN DATE '{chunk_start}' AND DATE '{chunk_end}'
    ) TO '{tmp_out}' (FORMAT PARQUET)
    """)
    tmp_out.rename(chunk_out)


for year in YEARS:
    year_out = OUT_DIR / f"edges_{year}.parquet"
    if year_out.exists():
        print(f"[{time.time()-t0:6.0f}s] year {year}: already done, skipping", flush=True)
        continue

    chunk_dir = OUT_DIR / f"_chunks_{year}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    month_cache_dir = OUT_DIR / f"_month_cache_{year}"

    Path(DB_FILE).unlink(missing_ok=True)
    con = duckdb.connect(DB_FILE)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET temp_directory='/tmp/duckdb_spill'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=1")  # single-threaded: transient multi-worker buffer spikes can
    # exceed memory_limit before DuckDB's own accounting catches up and spills

    load_year_raw_edges(con, year, "raw_edges", month_cache_dir)
    n_raw = con.execute("SELECT COUNT(*) FROM raw_edges").fetchone()[0]
    print(f"[{time.time()-t0:6.0f}s] year {year}: raw_edges loaded, {n_raw:,} rows", flush=True)

    import datetime as _dt
    chunk_start = _dt.date(year, 1, 1)
    year_end = _dt.date(year, 12, 31)
    n_chunks_done = 0
    n_chunks_skipped = 0
    while chunk_start <= year_end:
        chunk_end = min(chunk_start + _dt.timedelta(days=CHUNK_DAYS - 1), year_end)
        if VALIDATE and chunk_start > _dt.date(year, 2, 19):
            break

        chunk_out = chunk_dir / f"chunk_{chunk_start.isoformat()}.parquet"
        if chunk_out.exists():
            n_chunks_skipped += 1
            chunk_start += _dt.timedelta(days=CHUNK_DAYS)
            continue

        process_chunk(con, chunk_start, chunk_end, chunk_out)
        n_chunks_done += 1
        chunk_start += _dt.timedelta(days=CHUNK_DAYS)

    print(f"[{time.time()-t0:6.0f}s] year {year}: chunks complete "
          f"({n_chunks_done} new, {n_chunks_skipped} already done)", flush=True)

    con.close()
    Path(DB_FILE).unlink(missing_ok=True)

    # Concatenate all this year's chunk files into the single final output --
    # cheap and low-memory (reading already-small, already-filtered parquet
    # files), unlike the chunk computation itself.
    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    con2 = duckdb.connect()
    con2.execute(f"""
    COPY (SELECT * FROM read_parquet({[str(f) for f in chunk_files]}) ORDER BY date)
    TO '{year_out}' (FORMAT PARQUET)
    """)
    n_final = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{year_out}')").fetchone()[0]
    con2.close()
    print(f"[{time.time()-t0:6.0f}s] year {year}: saved {n_final:,} qualifying rows -> {year_out}", flush=True)

    if not VALIDATE:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        shutil.rmtree(month_cache_dir, ignore_errors=True)

    if VALIDATE:
        print(f"\n=== VALIDATION: Python reference gave 33,263 qualifying rows after 50 days (through {VALIDATE_END_DATE}) ===")
        print(f"DuckDB gives: {n_final:,} rows")
        print("MATCH" if n_final == 33263 else "MISMATCH -- do not trust this at full scale until reconciled")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        shutil.rmtree(month_cache_dir, ignore_errors=True)
        year_out.unlink(missing_ok=True)  # VALIDATE output isn't a real result, don't leave
        # it behind masquerading as a completed year (would wrongly skip the real run later)
        break

print("\n\nDONE.")
