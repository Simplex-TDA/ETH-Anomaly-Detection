"""
tad_ethereum_functions.py
=========================
All reusable functions for the Topological Anomaly Detection (TAD)
Ethereum pipeline.

Re-implementation of:
  Ofori-Boateng et al. (2021) — Topological Anomaly Detection in Dynamic
  Multilayer Blockchain Networks (arXiv:2106.01806)

Usage
-----
    from tad_ethereum_functions import *
"""

import warnings
warnings.filterwarnings('ignore')

import math
import time
import datetime
import json

import numpy as np
import pandas as pd
from pathlib import Path
# from tqdm.notebook import tqdm
from tqdm import tqdm

import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.stats import entropy

from ripser import ripser
from persim import wasserstein, bottleneck


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_filtered_graph(
    eth_dir: Path,
    erc20_dir: Path,
    global_file: Path,
    daily_dir: Path,
    filters: list,
    ranking_metric: str,
    global_top_num: int,
    daily_top_num: int,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Return a unified transaction DataFrame restricted to edges where at least
    one endpoint is a "top node" — either globally or on that specific day.

    Global top nodes
    ----------------
    An address qualifies if it appears in the global ranking file with:
      - filter in `filters`
      - ranking_metric == `ranking_metric`
      - bottom_rank <= global_top_num

    Daily top nodes
    ---------------
    An address qualifies on date D if it appears in the daily ranking file
    that covers D with:
      - filter in `filters`
      - ranking_metric == `ranking_metric`
      - bottom_rank <= daily_top_num
      - date == D

    An edge survives if BOTH its from_addr and to_addr are top nodes
    (global OR daily) on that day.

    Parameters
    ----------
    eth_dir         : folder with ETH-native parquets
    erc20_dir       : folder with ERC20 parquets
    global_file     : ranking/global_top_nodes.parquet
    daily_dir       : ranking/daily/
    filters         : list of filter keys to consider (e.g. ['contract_txs'])
    ranking_metric  : 'tx_count' or 'tx_value'
    global_top_num  : keep global nodes with bottom_rank <= this
    daily_top_num   : keep daily  nodes with bottom_rank <= this
    start_date      : restrict source data to on/after this date (optional)
    end_date        : restrict source data to on/before this date (optional)

    Returns
    -------
    DataFrame with all original columns from both ETH and ERC20 sources,
    plus an 'erc20' column ('ETH' for native transactions), filtered to
    qualifying edges only.
    Also prints average nodes and edges per day per filter.
    """

    # ── 1. Build global top-node set ─────────────────────────────────────────
    tqdm.write('[1/4] Loading global top nodes ...')
    global_nodes: set = set()

    if global_file.exists():
        gdf = pd.read_parquet(global_file)
        mask = (
            gdf['filter'].isin(filters) &
            (gdf['ranking_metric'] == ranking_metric) &
            (gdf['bottom_rank'] <= global_top_num)
        )
        global_nodes = set(gdf.loc[mask, 'address'].dropna().unique())
        tqdm.write(f'    {len(global_nodes):,} global top nodes '
                   f'(filters={filters}, metric={ranking_metric}, top={global_top_num})')
    else:
        tqdm.write(f'    [WARN] Global ranking file not found: {global_file}')

    # ── 2. Build daily top-node sets keyed by date ───────────────────────────
    tqdm.write('[2/4] Loading daily top nodes ...')

    daily_nodes: dict = {}
    daily_files = sorted(daily_dir.glob('*.parquet'))
    pbar = tqdm(daily_files, desc='  Daily ranking files', unit='file', leave=False)
    for fpath in pbar:
        pbar.set_postfix_str(fpath.name, refresh=False)
        ddf = pd.read_parquet(fpath)
        ddf['date'] = pd.to_datetime(ddf['date'])

        if start_date is not None:
            ddf = ddf[ddf['date'] >= start_date]
        if end_date is not None:
            ddf = ddf[ddf['date'] <= end_date]
        if ddf.empty:
            continue

        mask = (
            ddf['filter'].isin(filters) &
            (ddf['ranking_metric'] == ranking_metric) &
            (ddf['bottom_rank'] <= daily_top_num)
        )
        relevant = ddf.loc[mask, ['date', 'address']].dropna()

        for dt, grp in relevant.groupby('date', sort=False):
            dt = pd.Timestamp(dt)
            if dt not in daily_nodes:
                daily_nodes[dt] = set()
            daily_nodes[dt].update(grp['address'].unique())

    pbar.close()
    tqdm.write(f'    {len(daily_nodes):,} dates with daily top nodes')

    # ── 3. Stream source files and filter edges ───────────────────────────────
    tqdm.write('[3/4] Streaming source files ...')

    sources = [
        (eth_dir,   'ETH',  'ETH txs'),
        (erc20_dir,  None,  'ERC20 txs'),
    ]

    result_parts = []

    for folder, token_label, src_label in sources:
        files = sorted(folder.glob('*.parquet'))
        pbar  = tqdm(files, desc=f'  {src_label}', unit='file', leave=False)

        for fpath in pbar:
            pbar.set_postfix_str(fpath.name, refresh=False)
            chunk = pd.read_parquet(fpath)
            chunk['date'] = pd.to_datetime(chunk['date'])

            if token_label is not None:
                chunk['erc20'] = token_label

            if start_date is not None:
                chunk = chunk[chunk['date'] >= start_date]
            if end_date is not None:
                chunk = chunk[chunk['date'] <= end_date]
            if chunk.empty:
                continue

            frm = chunk['from_addr'].to_numpy(dtype=object)
            to  = chunk['to_addr'].to_numpy(dtype=object)
            dt  = chunk['date'].to_numpy()

            frm_global = pd.Series(frm).isin(global_nodes).to_numpy()
            to_global  = pd.Series(to).isin(global_nodes).to_numpy()

            frm_daily = np.zeros(len(frm), dtype=bool)
            to_daily  = np.zeros(len(to),  dtype=bool)

            for unique_dt in chunk['date'].unique():
                day_set = daily_nodes.get(pd.Timestamp(unique_dt), set())
                if not day_set:
                    continue
                row_mask = (dt == unique_dt)
                frm_daily[row_mask] = pd.Series(frm[row_mask]).isin(day_set).to_numpy()
                to_daily[row_mask]  = pd.Series(to[row_mask]).isin(day_set).to_numpy()

            frm_ok = frm_global | frm_daily
            to_ok  = to_global  | to_daily
            keep   = frm_ok & to_ok

            if keep.any():
                result_parts.append(chunk.loc[keep].copy())

        pbar.close()

    # ── 4. Combine and report ─────────────────────────────────────────────────
    tqdm.write('[4/4] Combining and computing statistics ...')

    if not result_parts:
        tqdm.write('    [WARN] No rows survived the filter.')
        return pd.DataFrame(columns=['date', 'from_addr', 'to_addr', 'erc20',
                                     'tx_count', 'tx_value', 'max_value',
                                     'total_gas_price', 'total_input_bytes',
                                     'min_input_bytes'])

    result = pd.concat(result_parts, ignore_index=True)
    tqdm.write(f'    Total rows: {len(result):,}')

    tqdm.write('\n  Per-filter daily averages:')
    tqdm.write(f'  {"Filter":<25} {"Avg edges/day":>15} {"Avg nodes/day":>15}')
    tqdm.write('  ' + '-' * 57)

    for f in filters:
        daily_stats = (
            result.groupby('date').apply(
                lambda g: pd.Series({
                    'edges': len(g),
                    'nodes': len(set(g['from_addr'].tolist()) | set(g['to_addr'].tolist()))
                })
            )
        )
        if daily_stats.empty:
            continue
        avg_edges = daily_stats['edges'].mean()
        avg_nodes = daily_stats['nodes'].mean()
        tqdm.write(f'  {f:<25} {avg_edges:>15.1f} {avg_nodes:>15.1f}')

    tqdm.write('')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(day_df, edge_weight_col='tx_count'):
    """
    Build an undirected weighted graph from one day's (and optionally one
    layer's) transactions.  The edge weight is taken from `edge_weight_col`.
    Multiple rows with the same (from, to) pair are summed.
    """
    G = nx.Graph()
    for _, row in day_df.iterrows():
        u = row['from_addr']
        v = row['to_addr']
        w = float(row[edge_weight_col])
        if w <= 0 or np.isnan(w):
            w = 1e-6
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def geodesic_densification(G, max_nodes=500, dist_function='norm_similarity', alpha=9):
    """
    Geodesic densification (Section 4 of the paper).

    1. Cap to top-N nodes by weighted degree
    2. Convert weight to distance:
       - d = 1 / log1p(w)  (more weight = closer) if dist_function = 'i/log'
       - 1 / (1 + alpha * (A_uv - A_min) / (A_max - A_min))
         if dist_function = 'norm_similarity'
    3. All-pairs shortest paths via Dijkstra
    4. Normalise to [0, 1]

    Returns (dist_matrix, node_list) or (None, []) if too sparse.
    """
    if G.number_of_nodes() == 0:
        return None, []

    if G.number_of_nodes() > max_nodes:
        top = sorted(G.degree(weight='weight'),
                     key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 3:
        return None, nodes

    idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, data = [], [], []

    if dist_function == 'norm_similarity':
        weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
        A_max = max(weights)
        A_min = min(weights)

    for u, v, d in G.edges(data=True):
        w = max(d.get('weight', 1e-9), 1e-9)
        if dist_function == 'norm_similarity':
            weight_range = A_max - A_min
            if weight_range == 0:
                dist = 1.0
            else:
                dist = 1 / (1 + alpha * (w - A_min) / weight_range)
        elif dist_function == 'i/log':
            dist = 1.0 / np.log1p(w)
        i, j = idx[u], idx[v]
        rows += [i, j];  cols += [j, i];  data += [dist, dist]

    sparse      = csr_matrix((data, (rows, cols)), shape=(n, n))
    dist_matrix = shortest_path(sparse, method='D', directed=False)

    finite = dist_matrix[np.isfinite(dist_matrix)]
    max_d  = finite.max() if len(finite) > 0 else 1.0
    dist_matrix[~np.isfinite(dist_matrix)] = 2.0 * max_d

    if dist_matrix.max() > 0:
        dist_matrix /= dist_matrix.max()

    return dist_matrix, nodes


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PERSISTENT HOMOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pd(dist_matrix, maxdim=1):
    """Run Vietoris-Rips persistent homology via Ripser."""
    result = ripser(dist_matrix, metric='precomputed', maxdim=maxdim)
    return result['dgms']


def dgms_to_finite(dgms):
    """
    Strip infinite-death entries and return {dim: array of finite (birth, death) pts}.
    Falls back to [[0,0]] per dimension so Wasserstein is always computable.
    """
    out = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            out[dim] = np.array([[0., 0.]])
            continue
        finite = dgm[dgm[:, 1] < np.inf]
        out[dim] = finite if len(finite) > 0 else np.array([[0., 0.]])
    return out


def stack_pds(layer_pd_dicts):
    """
    Stacked Persistence Diagram (Definition 3 of the paper).
    Concatenate finite persistence points across all layers, per dimension.
    Used only when MULTILAYER=True.
    """
    if not layer_pd_dicts:
        return {0: np.array([[0., 0.]]), 1: np.array([[0., 0.]])}

    all_dims = set()
    for pd_dict in layer_pd_dicts.values():
        all_dims |= set(pd_dict.keys())

    stacked = {}
    for dim in all_dims:
        pts = [pd_dict[dim] for pd_dict in layer_pd_dicts.values() if dim in pd_dict]
        stacked[dim] = np.vstack(pts) if pts else np.array([[0., 0.]])
    return stacked


def pd_distance(pd_a, pd_b, metric='wasserstein'):
    """
    Distance between two PD dicts, summed across all dimensions.
    D(PD_a, PD_b) = Σ_dim  metric(PD_a[dim], PD_b[dim])
    """
    total = 0.0
    for dim in set(pd_a) | set(pd_b):
        da = pd_a.get(dim, np.array([[0., 0.]]))
        db = pd_b.get(dim, np.array([[0., 0.]]))
        if len(da) == 0: da = np.array([[0., 0.]])
        if len(db) == 0: db = np.array([[0., 0.]])
        try:
            total += wasserstein(da, db) if metric == 'wasserstein' else bottleneck(da, db)
        except Exception:
            pass
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def persistence_features(distance_matrix, maxdim=1):
    """
    Extract topological features from a distance matrix using persistent homology.

    Parameters
    ----------
    distance_matrix : (n x n) numpy array
    maxdim          : max homology dimension (1 = includes loops)

    Returns
    -------
    dict of features
    """
    result = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim)
    diagrams = result['dgms']

    features = {}

    H0 = diagrams[0]
    H0_lifetimes = H0[:, 1] - H0[:, 0]
    H0_lifetimes = H0_lifetimes[np.isfinite(H0_lifetimes)]

    if len(diagrams) > 1:
        H1 = diagrams[1]
        H1_lifetimes = H1[:, 1] - H1[:, 0]
        H1_lifetimes = H1_lifetimes[np.isfinite(H1_lifetimes)]
    else:
        H1_lifetimes = np.array([])

    features['num_loops'] = len(H1_lifetimes)

    all_lifetimes = np.concatenate([H0_lifetimes, H1_lifetimes]) if len(H1_lifetimes) > 0 else H0_lifetimes
    features['avg_persistence'] = np.mean(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
    features['max_persistence'] = np.max(all_lifetimes) if len(all_lifetimes) > 0 else 0.0

    if len(all_lifetimes) > 0:
        probs = all_lifetimes / (np.sum(all_lifetimes) + 1e-10)
        features['persistence_entropy'] = entropy(probs)
    else:
        features['persistence_entropy'] = 0.0

    features['avg_loop_persistence'] = np.mean(H1_lifetimes) if len(H1_lifetimes) > 0 else 0.0
    features['max_loop_persistence'] = np.max(H1_lifetimes) if len(H1_lifetimes) > 0 else 0.0

    total_persistence = np.sum(all_lifetimes) + 1e-10
    loop_persistence  = np.sum(H1_lifetimes)
    features['loop_persistence_ratio'] = loop_persistence / total_persistence

    if len(H1_lifetimes) > 0:
        threshold = np.percentile(H1_lifetimes, 75)
        features['num_strong_loops'] = int(np.sum(H1_lifetimes > threshold))
    else:
        features['num_strong_loops'] = 0

    features['persistence_variance'] = np.var(all_lifetimes) if len(all_lifetimes) > 0 else 0.0

    return features


# ── Low-level helpers used by extract_daily_features ─────────────────────────

def _persistence_lifetimes(diagram):
    """Compute lifetimes (death - birth), ignoring inf."""
    return np.array([d - b for b, d in diagram if np.isfinite(d)])


def _total_persistence(diagram):
    return _persistence_lifetimes(diagram).sum()


def _max_persistence(diagram):
    lt = _persistence_lifetimes(diagram)
    return lt.max() if len(lt) > 0 else 0


def _num_features(diagram):
    return len(diagram)


def extract_daily_features(diagrams):
    """
    Build a tidy DataFrame of raw TDA features from the pd_series dict.

    Parameters
    ----------
    diagrams : dict[date -> {dim: np.array}]

    Returns
    -------
    pd.DataFrame  (one row per day)
    """
    rows = []
    for date in sorted(diagrams.keys()):
        H0 = diagrams[date][0]
        H1 = diagrams[date][1]
        rows.append({
            'date':                 date,
            'H0_num_components':    _num_features(H0),
            'H0_total_persistence': _total_persistence(H0),
            'H0_max_persistence':   _max_persistence(H0),
            'H1_num_loops':         _num_features(H1),
            'H1_total_persistence': _total_persistence(H1),
            'H1_max_persistence':   _max_persistence(H1),
        })
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def add_temporal_features(df, window=3):
    """Add rolling / change-rate temporal features to the features DataFrame."""
    df = df.copy()
    df['H1_persistence_change'] = df['H1_total_persistence'].diff()
    df['H1_loop_change']        = df['H1_num_loops'].diff()
    df['H1_persistence_mean']   = df['H1_total_persistence'].rolling(window).mean()
    df['H1_persistence_std']    = df['H1_total_persistence'].rolling(window).std()
    df['H1_autocorr'] = df['H1_total_persistence'].rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    return df


def compute_indices(df):
    """Compute connectivity, fragility, and reflexivity composite indices."""
    df = df.copy()
    for col in [
        'H0_total_persistence', 'H0_max_persistence',
        'H1_total_persistence', 'H1_max_persistence',
    ]:
        df[col + '_z'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    df['connectivity_index'] = (
        df['H0_total_persistence_z'] + df['H0_max_persistence_z']
    )
    df['fragility_index'] = -(
        df['H1_total_persistence_z'] + df['H1_max_persistence_z']
    )
    df['reflexivity_index'] = (
        df['H1_autocorr'].fillna(0) + df['H1_persistence_change'].fillna(0)
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN PIPELINE STEP: compute PD for one day
# ═══════════════════════════════════════════════════════════════════════════════

def compute_day_pd(day_df, edge_weight_col, layer_filters, max_nodes, maxdim,
                   dist_func='norm_similarity', alpha=9):
    """
    Compute the (stacked) persistence diagram for a single day.

    Parameters
    ----------
    day_df          : DataFrame slice for one day
    edge_weight_col : column name used as edge weight
    layer_filters   : dict {name: callable} or None
    max_nodes       : node cap per graph
    maxdim          : max homology dimension
    dist_func       : 'norm_similarity' or 'i/log'
    alpha           : constant in norm_similarity metric

    Returns
    -------
    pd_dict      : {dim: finite_pts} or None if not enough data
    info         : dict of diagnostic counts
    day_features : dict of per-layer persistence features (or None)
    """
    if layer_filters:
        layer_pd_dicts = {}
        info = {'n_layers_ok': 0}
        day_features = {}

        for layer_name, filt_fn in layer_filters.items():
            mask     = filt_fn(day_df)
            layer_df = day_df[mask]
            if len(layer_df) < 3:
                continue
            G = build_graph(layer_df, edge_weight_col)
            dist_matrix, nodes = geodesic_densification(G, max_nodes, dist_func, alpha)
            if dist_matrix is None or len(nodes) < 3:
                continue
            try:
                dgms = compute_pd(dist_matrix, maxdim)
                layer_pd_dicts[layer_name] = dgms_to_finite(dgms)
                info['n_layers_ok'] += 1
                day_features[layer_name] = persistence_features(dist_matrix, maxdim)
            except Exception:
                pass

        if not layer_pd_dicts:
            return None, info, None

        return stack_pds(layer_pd_dicts), info, day_features

    else:
        if len(day_df) < 3:
            return None, {}, None
        G = build_graph(day_df, edge_weight_col)
        dist_matrix, nodes = geodesic_densification(G, max_nodes, dist_func, alpha)
        if dist_matrix is None or len(nodes) < 3:
            return None, {'n_nodes': len(nodes)}, None
        try:
            dgms = compute_pd(dist_matrix, maxdim)
            day_features = persistence_features(dist_matrix, maxdim)
            return dgms_to_finite(dgms), {'n_nodes': len(nodes)}, day_features
        except Exception:
            return None, {}, None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  HIGH-LEVEL PIPELINE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tda_pipeline(df, edge_weight_col, layer_filters, max_nodes, maxdim,
                     similarity_metric, alpha):
    """
    Run the full T-step TDA loop over every day in `df`.

    Returns
    -------
    pd_series       : dict[date -> pd_dict]
    features_series : dict[date -> day_features]
    stats_df        : DataFrame of per-day diagnostics
    """
    dates = sorted(df['date'].unique())
    multilayer = bool(layer_filters)
    layer_names = list(layer_filters.keys()) if layer_filters else ['all']
    mode_label = f'multilayer ({len(layer_names)} layers)' if multilayer else 'single-layer'
    print(f'T-step: computing PDs in {mode_label} mode using edge weight "{edge_weight_col}"')

    pd_series       = {}
    features_series = {}
    day_stats       = []

    for date in tqdm(dates, desc='T-step: building PDs'):
        day_df = df[df['date'] == date]
        pd_dict, info, day_features = compute_day_pd(
            day_df,
            edge_weight_col=edge_weight_col,
            layer_filters=layer_filters,
            max_nodes=max_nodes,
            maxdim=maxdim,
            dist_func=similarity_metric,
            alpha=alpha,
        )
        if pd_dict is not None:
            pd_series[date]       = pd_dict
            features_series[date] = day_features
            day_stats.append({
                'date':   date,
                'h0_pts': len(pd_dict.get(0, [])),
                'h1_pts': len(pd_dict.get(1, [])),
                **info,
            })

    stats_df = pd.DataFrame(day_stats).set_index('date')
    print(f'\nPDs computed for {len(pd_series)} / {len(dates)} days')
    print(f'Avg H0 pts : {stats_df["h0_pts"].mean():.1f}')
    print(f'Avg H1 pts : {stats_df["h1_pts"].mean():.1f}')
    if multilayer:
        print(f'Avg layers with data/day : {stats_df["n_layers_ok"].mean():.1f} / {len(layer_names)}')

    return pd_series, features_series, stats_df


def run_distance_series(pd_series, distance_metric='wasserstein'):
    """
    Compute consecutive Wasserstein/Bottleneck distances across the pd_series.

    Returns
    -------
    pd.Series  indexed by date
    """
    sorted_dates   = sorted(pd_series.keys())
    wass_distances = []
    wass_dates     = []

    for i in tqdm(range(1, len(sorted_dates)), desc='Wasserstein distances'):
        d_prev = sorted_dates[i - 1]
        d_curr = sorted_dates[i]
        dist   = pd_distance(pd_series[d_prev], pd_series[d_curr], metric=distance_metric)
        wass_distances.append(dist)
        wass_dates.append(d_curr)

    distance_series = pd.Series(
        wass_distances,
        index=pd.DatetimeIndex(wass_dates),
        name='pd_distance',
    ).sort_index()

    print(f'Distance series: {len(distance_series)} points')
    print(f'  Mean : {distance_series.mean():.4f}')
    print(f'  Std  : {distance_series.std():.4f}')
    print(f'  Max  : {distance_series.max():.4f}  ({distance_series.idxmax().date()})')

    return distance_series


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SERIALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_json_serializable(obj):
    """Recursively convert numpy / pandas / datetime objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, pd.DataFrame):
        return {str(idx): make_json_serializable(row) for idx, row in obj.iterrows()}
    elif isinstance(obj, pd.Series):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.strftime('%Y-%m-%d')
    return obj


def replace_nan(obj):
    """Recursively replace NaN float values with None (JSON null)."""
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def save_run_results(file_path: Path, run_name: str, run: dict,
                     run_results: dict = None):
    """
    Serialise `run` and append/overwrite it in the JSON results file.

    Parameters
    ----------
    file_path    : path to the results JSON file
    run_name     : key for this run inside the JSON
    run          : dict with all run data
    run_results  : existing results dict (loaded externally); created fresh if None

    Returns
    -------
    Updated run_results dict (also written to disk).
    """
    if run_results is None:
        if file_path.exists():
            try:
                run_results = json.loads(file_path.read_text())
            except json.JSONDecodeError:
                run_results = {}
        else:
            run_results = {}

    run_results[run_name] = make_json_serializable(run)
    run_results = replace_nan(run_results)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(run_results, f, indent=2, ensure_ascii=False)

    print(f'Saved results to {file_path}  (runs: {list(run_results.keys())})')
    return run_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  BATCH RUNNER  —  multiple years × multiple layer-sets
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(
    years: list,
    layers: dict,
    tda_cfg: dict,
    path_cfg: dict,
):
    """
    Run the full pipeline for every combination of year and layer-set,
    saving one results JSON per year.

    Parameters
    ----------
    years : list of int
        E.g. [2023, 2024, 2025]

    layers : dict of {run_name: layer_filters_dict}
        Each key becomes the run_name (key) in the year's results file.
        Each value is a LAYER_FILTERS dict (or None for single-layer).
        Example::

            LAYERS = {
                'simple_and_contracts': {
                    'simple_txs':   lambda d: d['total_input_bytes'] <= 300,
                    'contract_txs': lambda d: d['total_input_bytes'] >  300,
                },
                'contracts_only': {
                    'contract_txs': lambda d: d['total_input_bytes'] >  300,
                },
            }

    tda_cfg : dict with keys:
        edge_weight_col   (str)
        max_nodes         (int)
        homology_maxdim   (int)
        distance_metric   (str)  'wasserstein' | 'bottleneck'
        similarity_metric (str)  'norm_similarity' | 'i/log'
        alpha             (int)
        ranking_metric    (str)
        global_top        (int)
        daily_top         (int)

    path_cfg : dict with keys:
        data_root         (Path)  root folder that contains per-year sub-folders
                                  structured as:
                                    data_root / YEAR / eth_tx_value_output/weekly
                                    data_root / YEAR / erc20_tx_value_output/weekly
        ranking_root      (Path)  root folder that contains per-year ranking dirs
                                  structured as:
                                    ranking_root / YEAR / global_top_nodes.parquet
                                    ranking_root / YEAR / daily/
        results_prefix    (str)   prefix for the output JSON filename
                                  e.g. 'run_results_V7' → 'run_results_V7_2024.json'

    Returns
    -------
    None  (results written to disk as side-effect)
    """
    data_root      = Path(path_cfg['data_root'])
    ranking_root   = Path(path_cfg['ranking_root'])
    results_prefix = path_cfg['results_prefix']

    total_runs = len(years) * len(layers)
    run_counter = 0

    for year in years:
        print(f'\n{"═"*70}')
        print(f'  YEAR {year}')
        print(f'{"═"*70}')

        # ── Per-year paths ────────────────────────────────────────────────────
        data_dir    = data_root / str(year)
        eth_dir     = data_dir / 'eth_tx_value_output/weekly'
        erc20_dir   = data_dir / 'erc20_tx_value_output/weekly'
        ranking_dir = ranking_root / str(year)
        global_file = ranking_dir / 'global_top_nodes.parquet'
        daily_dir   = ranking_dir / 'daily'
        results_file = Path(f'{results_prefix}_{year}.json')

        ranking_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)

        start_date = f'{year}-01-01'
        end_date   = f'{year}-12-31'

        for run_name, layer_filters in layers.items():
            run_counter += 1
            layer_names = list(layer_filters.keys()) if layer_filters else ['all']
            print(f'\n  [{run_counter}/{total_runs}] Run "{run_name}"'
                  f'  |  layers: {layer_names}')
            print(f'  {"-"*66}')

            # ── Step 1: Load data ─────────────────────────────────────────────
            t0 = time.perf_counter()
            filtered_df = load_filtered_graph(
                eth_dir        = eth_dir,
                erc20_dir      = erc20_dir,
                global_file    = global_file,
                daily_dir      = daily_dir,
                filters        = layer_names,
                ranking_metric = tda_cfg['ranking_metric'],
                global_top_num = tda_cfg['global_top'],
                daily_top_num  = tda_cfg['daily_top'],
                start_date     = start_date,
                end_date       = end_date,
            )
            filtering_time = (time.perf_counter() - t0) / 60

            if filtered_df.empty or 'date' not in filtered_df.columns:
                print(f'  [SKIP] No data for year={year}, run="{run_name}"')
                continue

            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            dates = sorted(filtered_df['date'].unique())

            if tda_cfg['edge_weight_col'] not in filtered_df.columns:
                print(f'  [SKIP] Edge weight column '
                      f'"{tda_cfg["edge_weight_col"]}" not found.')
                continue

            daily_edge_nums = (
                filtered_df[['date', 'from_addr']]
                .groupby('date').count()
                .rename(columns={'from_addr': 'edges'})
            )
            daily_node_nums = (
                filtered_df[['date', 'from_addr', 'to_addr']]
                .groupby('date').agg(
                    nodes=('from_addr',
                           lambda s: pd.concat(
                               [s, filtered_df.loc[s.index, 'to_addr']]
                           ).nunique())
                )
            )

            print(f'  Loaded {len(filtered_df):,} rows, {len(dates)} days  '
                  f'({filtering_time:.2f} min)')

            # ── Step 2: TDA pipeline ──────────────────────────────────────────
            t0 = time.perf_counter()
            pd_series, features_series, stats_df = run_tda_pipeline(
                df                = filtered_df,
                edge_weight_col   = tda_cfg['edge_weight_col'],
                layer_filters     = layer_filters,
                max_nodes         = tda_cfg['max_nodes'],
                maxdim            = tda_cfg['homology_maxdim'],
                similarity_metric = tda_cfg['similarity_metric'],
                alpha             = tda_cfg['alpha'],
            )
            tda_time = (time.perf_counter() - t0) / 60

            t0 = time.perf_counter()
            distance_series = run_distance_series(
                pd_series, distance_metric=tda_cfg['distance_metric']
            )
            distance_aggregation_time = (time.perf_counter() - t0) / 60

            # ── Step 3: Features + save ───────────────────────────────────────
            tda_index_features = compute_indices(
                add_temporal_features(extract_daily_features(pd_series))
            )

            config = {
                'layers':            layer_names,
                'year':              year,
                'max_nodes':         tda_cfg['max_nodes'],
                'homology_maxdim':   tda_cfg['homology_maxdim'],
                'distance_metric':   tda_cfg['distance_metric'],
                'edge_weight':       tda_cfg['edge_weight_col'],
                'global_top':        tda_cfg['global_top'],
                'daily_top':         tda_cfg['daily_top'],
                'ranking_metric':    tda_cfg['ranking_metric'],
                'similarity_metric': tda_cfg['similarity_metric'],
                'alpha':             tda_cfg['alpha'],
                'ranking_dir':       str(ranking_dir),
            }

            run = {
                'config':                    config,
                'num_edges':                 len(filtered_df),
                'num_nodes':                 pd.concat(
                    [filtered_df['from_addr'], filtered_df['to_addr']]
                ).nunique(),
                'daily_edge_nums':           daily_edge_nums,
                'daily_node_nums':           daily_node_nums,
                'pd_series':                 pd_series,
                'distance_series':           distance_series,
                'tda_index_features':        tda_index_features,
                'stats_df':                  stats_df,
                'features_series':           features_series,
                'layer_names':               layer_names,
                'filtering_time':            filtering_time,
                'tda_time':                  tda_time,
                'distance_aggregation_time': distance_aggregation_time,
            }

            save_run_results(results_file, run_name, run)
            print(f'  ✓  Saved "{run_name}" → {results_file}')

    print(f'\n{"═"*70}')
    print(f'  All {total_runs} runs complete.')
    print(f'{"═"*70}')


print('tad_ethereum_functions.py loaded ✓')
