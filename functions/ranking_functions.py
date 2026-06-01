"""
Ranking Functions
==================
ETH Transaction Graph — Global & Daily Top Node Ranking

Processes ETH and ERC20 transaction data, applies user-defined filters, builds
weighted graphs, and offers two ranking modes:

**Mode A — Edge-based ranking**
Selects the top-N edges by weight and collects the nodes connected to those edges.

**Mode B — Centrality-based ranking**
Ranks nodes directly by a graph centrality metric (default: PageRank).
Outputs the top-N central nodes per filter per period.
"""

import json
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce from_addr <= to_addr so (A->B) and (B->A) are the same undirected edge.
    Uses numpy object arrays for the swap to avoid dtype/None issues.
    """
    frm  = df['from_addr'].to_numpy(dtype=object)
    to   = df['to_addr'].to_numpy(dtype=object)
    swap = frm > to
    frm[swap], to[swap] = to[swap].copy(), frm[swap].copy()
    out = df.copy()
    out['from_addr'] = frm
    out['to_addr']   = to
    return out


def week_label(date: pd.Timestamp) -> str:
    """Days 1-7->W1, 8-14->W2, 15-21->W3, 22-28->W4, 29-31->W5."""
    return f'{date.strftime("%Y-%m")}_W{(date.day - 1) // 7 + 1}'


def filter_dead_addresses(chunk, dead_ads=[]):
    """Remove transactions involving dead/null addresses."""
    chunk = chunk[~(chunk['to_addr'].isin(dead_ads))]
    chunk = chunk[~(chunk['from_addr'].isin(dead_ads))]
    return chunk


def read_source_file(
    fpath: Path,
    token_label: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dead_ads: list
) -> pd.DataFrame:
    """
    Read one source parquet, restrict to [start, end], inject erc20='ETH' for
    ETH-native files (so filters referencing erc20 work), canonicalise edge
    direction, and return the full frame. Empty DataFrame if nothing in range.
    """
    chunk = pd.read_parquet(fpath)
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk = chunk[(chunk['date'] >= start) & (chunk['date'] <= end)]
    chunk = filter_dead_addresses(chunk, dead_ads)
    if chunk.empty:
        return chunk
    if token_label is not None:
        chunk['erc20'] = token_label
    return make_canonical(chunk)


def apply_filter_and_aggregate(chunk: pd.DataFrame, filter_fn, keep_cols) -> pd.DataFrame:
    """Apply user filter then aggregate to keep_cols — fully vectorised."""
    mask  = filter_fn(chunk)
    chunk = chunk.loc[mask, keep_cols]
    if chunk.empty:
        return chunk
    return (
        chunk.groupby(['date', 'from_addr', 'to_addr'], as_index=False, sort=False)
             [['tx_count', 'tx_value']].sum()
    )


def rank_nodes(edges: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    """
    Select top-N edges by metric; return [address, top_rank, bottom_rank].
    Works for both a single-day slice and a full-period aggregate.
    """
    if edges.empty:
        return pd.DataFrame(columns=['address', 'top_rank', 'bottom_rank'])

    top = (
        edges[['from_addr', 'to_addr', metric]]
        .nlargest(top_n, metric, keep='first')
        .reset_index(drop=True)
    )
    top['rank'] = top.index + 1   # 1-based

    node_ranks = pd.concat([
        top[['from_addr', 'rank']].rename(columns={'from_addr': 'address'}),
        top[['to_addr',   'rank']].rename(columns={'to_addr':   'address'}),
    ], ignore_index=True)

    node_ranks = node_ranks.dropna(subset=['address'])

    return (
        node_ranks.groupby('address', as_index=False)
                  .agg(top_rank=('rank', 'min'), bottom_rank=('rank', 'max'))
    )


def check_directories(eth_dir: Path, erc20_dir: Path) -> bool:
    """Sanity-check that data directories exist and contain parquet files."""
    ok = True
    for label, folder in [('ETH', eth_dir), ('ERC20', erc20_dir)]:
        files = sorted(folder.glob('*.parquet')) if folder.exists() else []
        if not folder.exists():
            print(f'[ERROR] {label} folder not found: {folder.resolve()}')
            ok = False
        elif not files:
            print(f'[ERROR] {label} folder exists but has no .parquet files: {folder.resolve()}')
            ok = False
        else:
            print(f'[OK]    {label}: {len(files)} parquet files in {folder.resolve()}')
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE FILE INDEX (CACHED)
# ═══════════════════════════════════════════════════════════════════════════

def _build_source_index(eth_dir: Path, erc20_dir: Path,
                        start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Scan source files and build week_label -> [(str_path, token_label)] map."""
    index: dict[str, list] = defaultdict(list)
    sources = [(eth_dir, 'ETH', 'ETH'), (erc20_dir, None, 'ERC20')]

    for folder, token_label, label in sources:
        files = sorted(folder.glob('*.parquet'))
        pbar  = tqdm(files, desc=f'  Indexing {label}', unit='file', leave=False)
        for fpath in pbar:
            pbar.set_postfix_str(fpath.name, refresh=False)
            try:
                dates = pd.read_parquet(fpath, columns=['date'])
                dates['date'] = pd.to_datetime(dates['date'])
                dates = dates.loc[
                    (dates['date'] >= start) & (dates['date'] <= end), 'date'
                ]
                if dates.empty:
                    continue
                for wl in dates.map(week_label).unique():
                    index[wl].append([str(fpath), token_label])
            except Exception as exc:
                tqdm.write(f'  [WARN] Could not index {fpath.name}: {exc}')
        pbar.close()

    return dict(index)


def get_source_index(eth_dir: Path, erc20_dir: Path, index_file: Path,
                     start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """
    Return the source-file index, loading from cache if available.
    The cache stores the date range it was built for; if it doesn't match
    the requested range the cache is ignored and rebuilt.
    """
    cache_key = {'start': str(start.date()), 'end': str(end.date())}

    if index_file.exists():
        with open(index_file) as f:
            cached = json.load(f)
        if cached.get('_meta') == cache_key:
            index_raw = {k: v for k, v in cached.items() if k != '_meta'}
            index = {
                wl: [(Path(fp), tok) for fp, tok in entries]
                for wl, entries in index_raw.items()
            }
            tqdm.write(
                f'[index] Loaded cache from {index_file}  '
                f'({len(index)} weeks, built for {cache_key["start"]} - {cache_key["end"]})'
            )
            return index
        else:
            tqdm.write(
                f'[index] Cache date range mismatch — rebuilding '
                f'(cached={cached.get("_meta")}, requested={cache_key})'
            )

    tqdm.write('[index] Building source-file index (this runs once, then is cached) ...')
    index_raw = _build_source_index(eth_dir, erc20_dir, start, end)

    to_save = {'_meta': cache_key}
    to_save.update(index_raw)
    with open(index_file, 'w') as f:
        json.dump(to_save, f, indent=2)
    tqdm.write(f'[index] Saved to {index_file}')

    return {
        wl: [(Path(fp), tok) for fp, tok in entries]
        for wl, entries in index_raw.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — GLOBAL RANKING (EDGE-BASED)
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_global(global_file: Path) -> pd.DataFrame:
    """Load existing global ranking file if it exists."""
    if global_file.exists():
        df = pd.read_parquet(global_file)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date']   = pd.to_datetime(df['end_date'])
        tqdm.write(f'[global] Loaded existing file — {len(df):,} rows.')
        return df
    tqdm.write('[global] No existing file — starting fresh.')
    return pd.DataFrame()


def global_already_computed(existing, filter_name, metric, start, end):
    """Check if a global ranking has already been computed."""
    if existing.empty:
        return False
    return (
        (existing['filter']         == filter_name) &
        (existing['ranking_metric'] == metric)      &
        (existing['start_date']     == start)       &
        (existing['end_date']       == end)
    ).any()


def run_global_ranking(filters, eth_dir: Path, erc20_dir: Path,
                       global_file: Path, metrics, keep_cols,
                       start, end, top_n, dead_ads):
    """
    Global edge-based ranking (top-N edges -> connected nodes).

    Saved after every filter so the job is safely restartable.
    """
    sources = [(eth_dir, 'ETH', 'ETH txs'), (erc20_dir, None, 'ERC20 txs')]

    filter_bar = tqdm(list(filters.items()), desc='[Global] Filters', unit='filter')
    for filter_name, filter_fn in filter_bar:
        filter_bar.set_postfix({'current': filter_name})

        existing = load_existing_global(global_file)

        pending = [m for m in metrics
                   if not global_already_computed(existing, filter_name, m, start, end)]
        if not pending:
            tqdm.write(f'[global] SKIP "{filter_name}" — already complete.')
            continue

        tqdm.write(f'\n[global] RUN "{filter_name}"  pending={pending}')

        acc_parts = []
        for folder, token_label, src_label in sources:
            files = sorted(folder.glob('*.parquet'))
            pbar  = tqdm(files, desc=f'  [{filter_name}] {src_label}',
                         unit='file', leave=False)
            for fpath in pbar:
                pbar.set_postfix_str(fpath.name, refresh=False)
                raw = read_source_file(fpath, token_label, start, end, dead_ads)
                if raw.empty:
                    continue
                agg = apply_filter_and_aggregate(raw, filter_fn, keep_cols)
                if not agg.empty:
                    acc_parts.append(agg[['from_addr', 'to_addr', 'tx_count', 'tx_value']])
            pbar.close()

        if not acc_parts:
            tqdm.write(f'[global]   No data survived the filter — skipping.')
            continue

        period_edges = (
            pd.concat(acc_parts, ignore_index=True)
              .groupby(['from_addr', 'to_addr'], as_index=False, sort=False)
              [['tx_count', 'tx_value']].sum()
        )
        del acc_parts
        tqdm.write(f'[global]   Period edge table: {len(period_edges):,} edges')

        new_records = []
        metric_bar = tqdm(pending, desc=f'  [{filter_name}] metrics',
                          unit='metric', leave=False)
        for metric in metric_bar:
            metric_bar.set_postfix({'metric': metric})
            nodes = rank_nodes(period_edges, metric, top_n)
            if nodes.empty:
                tqdm.write(f'[global]   [{metric}] No nodes — skipped.')
                continue
            nodes['filter']         = filter_name
            nodes['ranking_metric'] = metric
            nodes['start_date']     = start
            nodes['end_date']       = end
            new_records.append(
                nodes[['address', 'filter', 'ranking_metric',
                       'top_rank', 'bottom_rank', 'start_date', 'end_date']]
            )
            tqdm.write(f'[global]   [{metric}] {len(nodes):,} nodes ranked')
        metric_bar.close()
        del period_edges

        if new_records:
            combined = pd.concat([existing] + new_records, ignore_index=True)
            combined.to_parquet(global_file, index=False)
            tqdm.write(f'  ✓ [{filter_name}] saved -> {global_file}  ({len(combined):,} rows total)')

    filter_bar.close()
    tqdm.write(f'\n✓ Global ranking complete -> {global_file}')


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — DAILY RANKING (EDGE-BASED)
# ═══════════════════════════════════════════════════════════════════════════

def daily_ranking_path(daily_dir: Path, label: str) -> Path:
    """Return path for daily ranking file."""
    return daily_dir / f'daily_ranking_{label}.parquet'


def load_skip_index(daily_dir: Path, label: str) -> set:
    """
    Load the set of already-computed (filter, metric, date) triples for a week file.
    Reads only the three key columns to keep RAM minimal.
    """
    path = daily_ranking_path(daily_dir, label)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=['filter', 'ranking_metric', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    return set(zip(df['filter'], df['ranking_metric'], df['date']))


def run_daily_ranking(filters, eth_dir: Path, erc20_dir: Path,
                      daily_dir: Path, index_file: Path, metrics, keep_cols,
                      start, end, top_n, dead_ads):
    """
    Daily edge-based ranking — resumable, saves after every (filter, metric, day).
    """
    source_index = get_source_index(eth_dir, erc20_dir, index_file, start, end)
    all_weeks    = sorted(source_index.keys())
    tqdm.write(f'[daily] {len(all_weeks)} output weeks to process.')

    week_bar = tqdm(all_weeks, desc='[Daily] Weeks', unit='week')

    for wlabel in week_bar:
        week_bar.set_postfix_str(wlabel, refresh=False)

        year_s, rest = wlabel.split('-', 1)
        month_s, w_s = rest.split('_W')
        year_i, month_i, w_i = int(year_s), int(month_s), int(w_s)
        week_start = pd.Timestamp(year_i, month_i, (w_i - 1) * 7 + 1)
        week_end   = pd.Timestamp(
            year_i, month_i,
            min(w_i * 7, pd.Timestamp(year_i, month_i, 1).days_in_month)
        )
        week_start = max(week_start, start)
        week_end   = min(week_end,   end)

        raw_chunks = []
        src_bar = tqdm(
            source_index[wlabel],
            desc  = '  Reading sources',
            unit  = 'file',
            leave = False,
        )
        for fpath, token_label in src_bar:
            src_bar.set_postfix_str(fpath.name, refresh=False)
            raw = read_source_file(fpath, token_label, week_start, week_end, dead_ads)
            if not raw.empty:
                raw_chunks.append(raw)
        src_bar.close()

        if not raw_chunks:
            tqdm.write(f'  [{wlabel}] No data — skipping.')
            continue

        week_raw = pd.concat(raw_chunks, ignore_index=True)
        del raw_chunks

        any_new = False

        filter_bar = tqdm(
            list(filters.items()),
            desc  = f'  [{wlabel}] Filters',
            unit  = 'filter',
            leave = False,
        )
        for filter_name, filter_fn in filter_bar:
            filter_bar.set_postfix_str(filter_name, refresh=False)

            week_edges = apply_filter_and_aggregate(week_raw, filter_fn, keep_cols)

            if week_edges.empty:
                tqdm.write(f'  [{wlabel}/{filter_name}] No data after filter.')
                continue

            week_dates = sorted(week_edges['date'].unique())

            skip = load_skip_index(daily_dir, wlabel)

            metric_bar = tqdm(
                metrics,
                desc  = f'    [{filter_name}] Metrics',
                unit  = 'metric',
                leave = False,
            )
            for metric in metric_bar:
                metric_bar.set_postfix_str(metric, refresh=False)

                day_bar = tqdm(
                    week_dates,
                    desc  = f'      [{metric}] Days',
                    unit  = 'day',
                    leave = False,
                )
                for d in day_bar:
                    dt = pd.Timestamp(d)
                    day_bar.set_postfix_str(str(dt.date()), refresh=False)

                    if (filter_name, metric, dt) in skip:
                        continue

                    day_edges = week_edges[week_edges['date'] == d]
                    nodes     = rank_nodes(day_edges, metric, top_n)

                    if nodes.empty:
                        skip.add((filter_name, metric, dt))
                        continue

                    nodes['filter']         = filter_name
                    nodes['ranking_metric'] = metric
                    nodes['date']           = dt
                    day_row = nodes[[
                        'address', 'filter', 'ranking_metric',
                        'top_rank', 'bottom_rank', 'date',
                    ]]

                    out_path = daily_ranking_path(daily_dir, wlabel)
                    if out_path.exists():
                        existing = pd.read_parquet(out_path)
                        day_row  = pd.concat([existing, day_row], ignore_index=True)
                        del existing
                    day_row.to_parquet(out_path, index=False)
                    del day_row

                    skip.add((filter_name, metric, dt))
                    any_new = True

                day_bar.close()
            metric_bar.close()
            del week_edges

        filter_bar.close()
        del week_raw

        if any_new:
            n = len(pd.read_parquet(daily_ranking_path(daily_dir, wlabel), columns=['date']))
            tqdm.write(f'  ✓ {wlabel} saved ({n:,} rows)')

    week_bar.close()
    tqdm.write(f'\n✓ Daily ranking complete -> {daily_dir.resolve()}')


# ═══════════════════════════════════════════════════════════════════════════
# CENTRALITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Number of sampled sources for approximated betweenness
BETWEENNESS_K = 200


def _hits_hub(G, w):
    """Return hub scores from HITS algorithm."""
    hubs, _ = nx.hits(G, max_iter=300, normalized=True)
    return hubs


def _hits_auth(G, w):
    """Return authority scores from HITS algorithm."""
    _, auths = nx.hits(G, max_iter=300, normalized=True)
    return auths


def _k_core_scores(G, w):
    """Return k-core numbers (unweighted by definition)."""
    return nx.core_number(G)


def _eigenvector_safe(G, w):
    """Eigenvector centrality with fallback to numpy if iterative fails."""
    try:
        return nx.eigenvector_centrality(G, weight=w, max_iter=500, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        return nx.eigenvector_centrality_numpy(G, weight=w)


def _clustering(G, w):
    """Weighted clustering coefficient."""
    return nx.clustering(G, weight=w)


# Centrality function registry
CENTRALITY_FNS = {
    'page_rank':          lambda G, w: nx.pagerank(G, weight=w, alpha=0.85, max_iter=300),
    'degree':             lambda G, w: dict(G.degree()),
    'strength':           lambda G, w: dict(G.degree(weight=w)),
    'k_core':             _k_core_scores,
    'betweenness_approx': lambda G, w: nx.betweenness_centrality(
                               G, weight=w, normalized=True, k=BETWEENNESS_K),
    'hits_hub':           _hits_hub,
    'hits_authority':     _hits_auth,
    'eigenvector':        _eigenvector_safe,
    'clustering':         _clustering,
}


def build_graph(edges: pd.DataFrame, weight_col: str,
                subgraph_top_n: int | None) -> nx.Graph:
    """
    Build an undirected weighted NetworkX graph from an edge DataFrame.
    Parallel edges are summed. Optionally prune to top-N nodes by degree.
    Self-loops are removed.
    """
    agg = (
        edges.groupby(['from_addr', 'to_addr'], as_index=False, sort=False)
             [weight_col].sum()
    )
    agg = agg[agg['from_addr'] != agg['to_addr']]
    G = nx.Graph()
    G.add_weighted_edges_from(
        zip(agg['from_addr'], agg['to_addr'], agg[weight_col]),
        weight=weight_col,
    )
    if subgraph_top_n is not None and G.number_of_nodes() > subgraph_top_n:
        top_nodes = {n for n, _ in
                     sorted(G.degree(), key=lambda x: x[1], reverse=True)
                     [:subgraph_top_n]}
        G = G.subgraph(top_nodes).copy()
    return G


def _edge_scores_betweenness(G: nx.Graph, weight_col: str) -> dict:
    """Edge betweenness centrality (k-sampled approximation)."""
    raw = nx.edge_betweenness_centrality(G, weight=weight_col,
                                          normalized=True, k=BETWEENNESS_K)
    return {(min(u, v), max(u, v)): s for (u, v), s in raw.items()}


def _edge_scores_projection(G: nx.Graph, node_scores: dict,
                             agg: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """
    Projection fallback: edge score = max(score_from, score_to).
    """
    agg = agg.copy()
    agg['edge_score'] = (
        agg['from_addr'].map(node_scores).fillna(0)
        .combine(agg['to_addr'].map(node_scores).fillna(0), max)
    )
    return agg


def centrality_rank_nodes(edges: pd.DataFrame, centrality_metric: str,
                           top_n: int, weight_col: str,
                           subgraph_top_n: int | None) -> pd.DataFrame:
    """
    Rank nodes (or nodes connected to top edges) by a centrality metric.

    NODE MODE: metric name is a key in CENTRALITY_FNS
    EDGE MODE: metric name ends with "_edge"
    """
    if edges.empty:
        return pd.DataFrame(columns=['address', 'top_rank', 'bottom_rank'])

    is_edge_mode = centrality_metric.endswith('_edge')
    base_metric  = centrality_metric[:-5] if is_edge_mode else centrality_metric

    if base_metric not in CENTRALITY_FNS:
        available = list(CENTRALITY_FNS) + [m + '_edge' for m in CENTRALITY_FNS]
        raise ValueError(
            f"Unknown centrality metric '{centrality_metric}'. "
            f"Available: {available}"
        )

    G = build_graph(edges, weight_col, subgraph_top_n)

    kept_nodes = set(G.nodes())
    agg = (
        edges.groupby(['from_addr', 'to_addr'], as_index=False, sort=False)
             [weight_col].sum()
    )
    if subgraph_top_n is not None:
        agg = agg[agg['from_addr'].isin(kept_nodes) &
                  agg['to_addr'].isin(kept_nodes)]

    if not is_edge_mode:
        # Node mode
        scores = pd.Series(CENTRALITY_FNS[base_metric](G, weight_col),
                           name='score').sort_values(ascending=False)
        top = scores.head(top_n).reset_index().rename(columns={'index': 'address'})
        top['top_rank']    = range(1, len(top) + 1)
        top['bottom_rank'] = top['top_rank']
        return top[['address', 'top_rank', 'bottom_rank']]

    else:
        # Edge mode
        if base_metric == 'betweenness_approx':
            eb = _edge_scores_betweenness(G, weight_col)
            agg['edge_score'] = agg.apply(
                lambda r: eb.get((min(r['from_addr'], r['to_addr']),
                                   max(r['from_addr'], r['to_addr'])), 0.0),
                axis=1
            )
        else:
            node_scores = CENTRALITY_FNS[base_metric](G, weight_col)
            agg = _edge_scores_projection(G, node_scores, agg, weight_col)

        top_edges = (
            agg.nlargest(top_n, 'edge_score', keep='first')
               .reset_index(drop=True)
        )
        top_edges['rank'] = top_edges.index + 1

        node_ranks = pd.concat([
            top_edges[['from_addr', 'rank']].rename(columns={'from_addr': 'address'}),
            top_edges[['to_addr',   'rank']].rename(columns={'to_addr':   'address'}),
        ], ignore_index=True).dropna(subset=['address'])

        return (
            node_ranks.groupby('address', as_index=False)
                      .agg(top_rank=('rank', 'min'),
                           bottom_rank=('rank', 'max'))
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 1B — GLOBAL CENTRALITY RANKING
# ═══════════════════════════════════════════════════════════════════════════

def run_global_centrality_ranking(
    filters, eth_dir: Path, erc20_dir: Path, global_file: Path,
    centrality_metrics, start, end, top_n, dead_ads,
    weight_col, subgraph_top_n
):
    """
    Global centrality ranking — saved after every (filter, metric) pair.
    """
    sources = [(eth_dir, 'ETH', 'ETH txs'), (erc20_dir, None, 'ERC20 txs')]

    filter_bar = tqdm(list(filters.items()), desc='[Global-Centrality] Filters',
                      unit='filter')
    for filter_name, filter_fn in filter_bar:
        filter_bar.set_postfix({'current': filter_name})
        existing = load_existing_global(global_file)

        pending = [m for m in centrality_metrics
                   if not global_already_computed(existing, filter_name, m, start, end)]
        if not pending:
            tqdm.write(f'[global-centrality] SKIP "{filter_name}" — already complete.')
            continue

        tqdm.write(f'\n[global-centrality] RUN "{filter_name}"  pending={pending}')

        acc_parts = []
        for folder, token_label, src_label in sources:
            files = sorted(folder.glob('*.parquet'))
            pbar  = tqdm(files, desc=f'  [{filter_name}] {src_label}',
                         unit='file', leave=False)
            for fpath in pbar:
                pbar.set_postfix_str(fpath.name, refresh=False)
                raw = read_source_file(fpath, token_label, start, end, dead_ads)
                if raw.empty:
                    continue
                agg = apply_filter_and_aggregate(raw, filter_fn, ['date', 'from_addr', 'to_addr', 'tx_count', 'tx_value'])
                if not agg.empty:
                    acc_parts.append(agg[['from_addr', 'to_addr', weight_col]])
            pbar.close()

        if not acc_parts:
            tqdm.write(f'[global-centrality]   No data survived the filter — skipping.')
            continue

        period_edges = (
            pd.concat(acc_parts, ignore_index=True)
              .groupby(['from_addr', 'to_addr'], as_index=False, sort=False)
              [[weight_col]].sum()
        )
        del acc_parts
        n_nodes = period_edges['from_addr'].nunique() + period_edges['to_addr'].nunique()
        tqdm.write(f'[global-centrality]   {len(period_edges):,} edges, ~{n_nodes:,} nodes')
        if subgraph_top_n:
            tqdm.write(f'[global-centrality]   Will prune to top {subgraph_top_n:,} nodes by degree')

        for cm in tqdm(pending, desc=f'  [{filter_name}] centrality', unit='metric', leave=False):
            tqdm.write(f'[global-centrality]   Computing {cm} ...')
            try:
                nodes = centrality_rank_nodes(
                    period_edges, cm, top_n, weight_col, subgraph_top_n)
            except Exception as exc:
                tqdm.write(f'[global-centrality]   [{cm}] FAILED: {exc}')
                continue

            if nodes.empty:
                tqdm.write(f'[global-centrality]   [{cm}] No nodes — skipped.')
                continue

            nodes['filter']         = filter_name
            nodes['ranking_metric'] = cm
            nodes['start_date']     = start
            nodes['end_date']       = end

            existing = load_existing_global(global_file)
            combined = pd.concat(
                [existing,
                 nodes[['address', 'filter', 'ranking_metric',
                         'top_rank', 'bottom_rank', 'start_date', 'end_date']]],
                ignore_index=True
            )
            combined.to_parquet(global_file, index=False)
            tqdm.write(f'  ✓ [{filter_name}/{cm}] {len(nodes):,} nodes saved')

        del period_edges

    filter_bar.close()
    tqdm.write(f'\n✓ Global centrality ranking complete -> {global_file}')


# ═══════════════════════════════════════════════════════════════════════════
# PART 2B — DAILY CENTRALITY RANKING
# ═══════════════════════════════════════════════════════════════════════════

def daily_centrality_ranking_path(daily_dir: Path, label: str) -> Path:
    """Return path for daily centrality ranking file."""
    return daily_dir / f'daily_centrality_{label}.parquet'


def load_centrality_skip_index(daily_dir: Path, label: str) -> set:
    """Load skip index for centrality ranking."""
    path = daily_centrality_ranking_path(daily_dir, label)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=['filter', 'ranking_metric', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    return set(zip(df['filter'], df['ranking_metric'], df['date']))


def run_daily_centrality_ranking(
    filters, eth_dir: Path, erc20_dir: Path, daily_dir: Path,
    index_file: Path, centrality_metrics, start, end, top_n,
    dead_ads, weight_col, subgraph_top_n
):
    """
    Daily centrality ranking — resumable, chunked design.
    """
    source_index = get_source_index(eth_dir, erc20_dir, index_file, start, end)
    all_weeks    = sorted(source_index.keys())
    tqdm.write(f'[daily-centrality] {len(all_weeks)} output weeks to process.')
    if subgraph_top_n:
        tqdm.write(f'[daily-centrality] Subgraph pruning: top {subgraph_top_n:,} nodes by degree per day')

    week_bar = tqdm(all_weeks, desc='[Daily-Centrality] Weeks', unit='week')

    for wlabel in week_bar:
        week_bar.set_postfix_str(wlabel, refresh=False)

        year_s, rest   = wlabel.split('-', 1)
        month_s, w_s   = rest.split('_W')
        year_i, month_i, w_i = int(year_s), int(month_s), int(w_s)
        week_start = pd.Timestamp(year_i, month_i, (w_i - 1) * 7 + 1)
        week_end   = pd.Timestamp(
            year_i, month_i,
            min(w_i * 7, pd.Timestamp(year_i, month_i, 1).days_in_month)
        )
        week_start = max(week_start, start)
        week_end   = min(week_end,   end)

        raw_chunks = []
        src_bar = tqdm(source_index[wlabel], desc='  Reading sources',
                       unit='file', leave=False)
        for fpath, token_label in src_bar:
            src_bar.set_postfix_str(fpath.name, refresh=False)
            raw = read_source_file(fpath, token_label, week_start, week_end, dead_ads)
            if not raw.empty:
                raw_chunks.append(raw)
        src_bar.close()

        if not raw_chunks:
            tqdm.write(f'  [{wlabel}] No data — skipping.')
            continue

        week_raw = pd.concat(raw_chunks, ignore_index=True)
        del raw_chunks

        any_new = False

        filter_bar = tqdm(list(filters.items()), desc=f'  [{wlabel}] Filters',
                          unit='filter', leave=False)
        for filter_name, filter_fn in filter_bar:
            filter_bar.set_postfix_str(filter_name, refresh=False)

            week_edges = apply_filter_and_aggregate(week_raw, filter_fn,
                                                    ['date', 'from_addr', 'to_addr', 'tx_count', 'tx_value'])
            if week_edges.empty:
                tqdm.write(f'  [{wlabel}/{filter_name}] No data after filter.')
                continue

            week_dates = sorted(week_edges['date'].unique())
            skip = load_centrality_skip_index(daily_dir, wlabel)

            cm_bar = tqdm(list(centrality_metrics),
                          desc=f'    [{filter_name}] Centrality',
                          unit='metric', leave=False)
            for cm in cm_bar:
                cm_bar.set_postfix_str(cm, refresh=False)

                day_bar = tqdm(week_dates, desc=f'      [{cm}] Days',
                               unit='day', leave=False)
                for d in day_bar:
                    dt = pd.Timestamp(d)
                    day_bar.set_postfix_str(str(dt.date()), refresh=False)

                    if (filter_name, cm, dt) in skip:
                        continue

                    day_edges = week_edges[week_edges['date'] == d]
                    try:
                        nodes = centrality_rank_nodes(
                            day_edges, cm, top_n, weight_col, subgraph_top_n)
                    except Exception as exc:
                        tqdm.write(
                            f'  [{wlabel}/{filter_name}/{cm}/{dt.date()}] '
                            f'FAILED: {exc} — skipping day.')
                        skip.add((filter_name, cm, dt))
                        continue

                    if nodes.empty:
                        skip.add((filter_name, cm, dt))
                        continue

                    nodes['filter']         = filter_name
                    nodes['ranking_metric'] = cm
                    nodes['date']           = dt
                    day_row = nodes[[
                        'address', 'filter', 'ranking_metric',
                        'top_rank', 'bottom_rank', 'date',
                    ]]

                    out_path = daily_centrality_ranking_path(daily_dir, wlabel)
                    if out_path.exists():
                        existing_df = pd.read_parquet(out_path)
                        day_row     = pd.concat([existing_df, day_row], ignore_index=True)
                        del existing_df
                    day_row.to_parquet(out_path, index=False)
                    del day_row

                    skip.add((filter_name, cm, dt))
                    any_new = True

                day_bar.close()
            cm_bar.close()
            del week_edges

        filter_bar.close()
        del week_raw

        if any_new:
            n = len(pd.read_parquet(daily_centrality_ranking_path(daily_dir, wlabel),
                                    columns=['date']))
            tqdm.write(f'  ✓ {wlabel} saved ({n:,} rows)')

    week_bar.close()
    tqdm.write(f'\n✓ Daily centrality ranking complete -> {daily_dir.resolve()}')


print('ranking_functions.py loaded ✓')
