"""
period_analysis_functions.py
=============================
Functions for multi-year continuous Wasserstein series analysis:
  - Loading per-year result files
  - Stitching cross-year boundaries
  - Change-point detection (multivariate, ruptures)
  - Period characterisation
  - Anomaly detection (S-ESD, IQR, Z-score, Rolling σ, Isolation Forest)
  - Saving results to JSON
"""

import warnings, json, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

from tad_ethereum_functions import pd_distance, make_json_serializable, replace_nan


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_year_results(years, file_pattern, runs):
    """
    Load the per-year JSON files and return nested dicts:
      year_data[year][run_name] = run_dict

    Keys extracted per run:
      pd_series       : dict[date_str -> {dim_str -> [[b,d], ...]}]
      distance_series : dict[date_str -> float]
    """
    year_data = {}
    for year in years:
        fpath = Path(file_pattern.format(year=year))
        if not fpath.exists():
            print(f'  [WARN] {fpath} not found — skipping year {year}')
            continue
        with open(fpath, 'r') as f:
            raw = json.load(f)
        year_data[year] = {}
        for run in runs:
            if run not in raw:
                print(f'  [WARN] Run "{run}" not found in {fpath}')
                continue
            year_data[year][run] = raw[run]
        print(f'  {fpath.name}  →  runs: {list(year_data[year].keys())}')
    return year_data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STITCHING
# ═══════════════════════════════════════════════════════════════════════════════

def pd_series_from_run(run_dict):
    """
    Reconstruct pd_series = {pd.Timestamp -> {int_dim: np.array}}
    from the JSON-serialised run dict.
    Integer dim keys were stringified by JSON; we restore them.
    """
    raw = run_dict.get('pd_series', {})
    out = {}
    for date_str, dim_dict in raw.items():
        ts = pd.Timestamp(date_str)
        out[ts] = {int(k): np.array(v) for k, v in dim_dict.items()}
    return out


def distance_series_from_run(run_dict):
    """
    Reconstruct distance_series = pd.Series(float, index=DatetimeIndex)
    from the JSON-serialised run dict.
    """
    raw = run_dict.get('distance_series', {})
    if not raw:
        return pd.Series(dtype=float)
    dates  = pd.DatetimeIndex([pd.Timestamp(d) for d in raw.keys()])
    values = list(raw.values())
    return pd.Series(values, index=dates, dtype=float).sort_index()


def stitch_distance_series(run_name, year_data, loaded_years):
    """
    Build a gapless daily Wasserstein series spanning all loaded years.

    Algorithm
    ---------
    1. Collect within-year distance series for every year that has this run.
    2. For each consecutive year-pair (Y, Y+1) where both have this run:
       a. Find the last date with a PD in year Y  (usually Dec 31)
       b. Find the first date with a PD in year Y+1 (usually Jan 1)
       c. Compute W₁(PD_last_Y, PD_first_{Y+1}) and assign it to the
          first date of Y+1.
    3. Concatenate and sort.

    Returns
    -------
    pd.Series  — daily Wasserstein distances, sorted by date
    list       — dates that were cross-year stitched (for bookkeeping)
    """
    parts = []
    stitched_dates = []

    relevant_years = [y for y in loaded_years
                      if run_name in year_data.get(y, {})]

    if not relevant_years:
        print(f'  [WARN] No data for run "{run_name}"')
        return pd.Series(dtype=float), []

    for year in relevant_years:
        s = distance_series_from_run(year_data[year][run_name])
        if not s.empty:
            parts.append(s)

    for ya, yb in zip(relevant_years[:-1], relevant_years[1:]):
        if yb != ya + 1:
            print(f'  [INFO] Non-consecutive years {ya}→{yb}, skipping boundary stitch')
            continue

        pds_a = pd_series_from_run(year_data[ya][run_name])
        pds_b = pd_series_from_run(year_data[yb][run_name])

        if not pds_a or not pds_b:
            continue

        last_date_a  = max(pds_a.keys())
        first_date_b = min(pds_b.keys())

        try:
            cross_dist = pd_distance(pds_a[last_date_a], pds_b[first_date_b])
            boundary_series = pd.Series(
                [cross_dist],
                index=pd.DatetimeIndex([first_date_b]),
                dtype=float,
            )
            parts.append(boundary_series)
            stitched_dates.append(str(first_date_b.date()))
            print(f'  Stitched {ya}→{yb}:  '
                  f'{last_date_a.date()} → {first_date_b.date()}  '
                  f'dist={cross_dist:.4f}')
        except Exception as e:
            print(f'  [WARN] Stitching {ya}→{yb} failed: {e}')

    if not parts:
        return pd.Series(dtype=float), stitched_dates

    combined = pd.concat(parts).sort_index()
    combined = combined.groupby(combined.index).max()
    return combined, stitched_dates


def stitch_all_runs(runs, year_data, loaded_years):
    """
    Stitch distance series for every run.

    Returns
    -------
    all_series : dict  run_name -> pd.Series
    stitch_log : dict  run_name -> list of stitched date strings
    """
    all_series = {}
    stitch_log = {}

    for run_name in runs:
        print(f'\nStitching run: "{run_name}"')
        s, sdates = stitch_distance_series(run_name, year_data, loaded_years)
        if s.empty:
            print(f'  → empty — skipping')
            continue
        all_series[run_name] = s
        stitch_log[run_name] = sdates
        print(f'  → {len(s)} daily points  '
              f'({s.index.min().date()} – {s.index.max().date()})  '
              f'{len(sdates)} boundary point(s) stitched')

    print(f'\nContinuous series ready for {len(all_series)} run(s).')
    return all_series, stitch_log


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHANGE-POINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_changepoints(series_dict, cp_min, cp_max, model='l2'):
    """
    Detect change points treating all runs as a multivariate signal.

    PELT's cp-count function is piecewise-constant and can jump over values
    (e.g. go directly 4→10 with no penalty giving 5-7).  The strategy:
      1. Walk a coarse grid high→low until the count first exceeds cp_max.
      2. We now have a bracket [pen_above, pen_below] where the count goes
         from ≤cp_max to >cp_max.  Search that bracket densely (100 points)
         to find the widest possible window that gives a count in [cp_min,cp_max].
      3. Fall back to the closest result if none found.
    """
    df = pd.DataFrame(series_dict)
    df = df.sort_index().interpolate(method='time').ffill().bfill().dropna()
    scaler = StandardScaler()
    signal = scaler.fit_transform(df.values).astype(np.float64)
    T = len(signal)
    print(f'  Signal shape: {signal.shape}  (T={T}, runs={list(df.columns)})')

    algo = rpt.Pelt(model=model, min_size=90, jump=1).fit(signal)

    target_mid   = (cp_min + cp_max) / 2.0
    best_bkps    = None
    closest_bkps = None
    closest_dist = float('inf')

    def _try(pen):
        nonlocal closest_bkps, closest_dist
        bkps = algo.predict(pen=pen)
        n = len(bkps) - 1
        d = abs(n - target_mid)
        if d < closest_dist:
            closest_dist  = d
            closest_bkps  = bkps
        return n, bkps

    coarse   = np.geomspace(200, 0.5, 40)
    pen_above = None
    n_above   = None
    pen_below = None

    for pen in coarse:
        n, bkps = _try(pen)
        print(f'  pen={pen:7.3f}  →  {n} cp(s)')

        if cp_min <= n <= cp_max:
            best_bkps = bkps
            print(f'  ✓ Found in range on coarse grid')
            break

        if n <= cp_max:
            pen_above = pen
            n_above   = n
        else:
            pen_below = pen
            break

    if best_bkps is None and pen_above is not None and pen_below is not None:
        print(f'  Bracket found: pen [{pen_below:.3f}, {pen_above:.3f}]  '
              f'(counts went from {n_above} to > {cp_max}). Dense search...')
        dense = np.geomspace(pen_above, pen_below, 80)
        for pen in dense:
            n, bkps = _try(pen)
            print(f'    pen={pen:7.3f}  →  {n} cp(s)')
            if cp_min <= n <= cp_max:
                best_bkps = bkps
                print(f'  ✓ Found in range in dense search')
                break

    if best_bkps is None:
        n_cl = len(closest_bkps) - 1
        print(f'  [WARN] Target range [{cp_min},{cp_max}] is not achievable '
              f'(PELT jumps over it). Using closest: {n_cl} cp(s)')
        best_bkps = closest_bkps

    cp_indices = [b for b in best_bkps if b < T]
    cp_dates   = [df.index[i] for i in cp_indices]
    print(f'  → {len(cp_dates)} change point(s):')
    for cp in cp_dates:
        print(f'      {cp.date()}')
    return cp_dates, len(cp_dates), df


def build_periods(cp_dates, signal_df):
    """Convert change-point dates + signal index into (start, end) period tuples."""
    all_dates  = signal_df.index
    boundaries = [all_dates[0]] + cp_dates + [all_dates[-1]]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PERIOD CHARACTERISATION
# ═══════════════════════════════════════════════════════════════════════════════

def characterise_period(series_slice):
    """
    Compute descriptive statistics for one period's distance slice.
    Returns a dict of scalar stats.
    """
    x = series_slice.dropna().values
    if len(x) == 0:
        return {}
    return {
        'n_days':        int(len(x)),
        'mean':          float(np.mean(x)),
        'median':        float(np.median(x)),
        'std':           float(np.std(x)),
        'variance':      float(np.var(x)),
        'min':           float(np.min(x)),
        'max':           float(np.max(x)),
        'range':         float(np.max(x) - np.min(x)),
        'skewness':      float(scipy_stats.skew(x)),
        'kurtosis':      float(scipy_stats.kurtosis(x)),
        'iqr':           float(scipy_stats.iqr(x)),
        'cv':            float(np.std(x) / (np.mean(x) + 1e-12)),
        'p5':            float(np.percentile(x, 5)),
        'p25':           float(np.percentile(x, 25)),
        'p75':           float(np.percentile(x, 75)),
        'p95':           float(np.percentile(x, 95)),
        'autocorr_lag1': float(pd.Series(x).autocorr(lag=1)) if len(x) > 2 else float('nan'),
        'autocorr_lag7': float(pd.Series(x).autocorr(lag=7)) if len(x) > 8 else float('nan'),
        'trend_slope':   float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0,
    }


def characterise_all_periods(all_series, periods):
    """
    Compute period stats for every run × every period.

    Returns
    -------
    dict  run_name -> list of period stat dicts (one per period)
    """
    period_stats = {}
    for run_name, series in all_series.items():
        period_stats[run_name] = []
        for i, (p_start, p_end) in enumerate(periods):
            mask  = (series.index >= p_start) & (series.index <= p_end)
            slc   = series[mask]
            stats = characterise_period(slc)
            stats['period_index'] = i + 1
            stats['start']        = str(p_start.date())
            stats['end']          = str(p_end.date())
            period_stats[run_name].append(stats)
    return period_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _grubbs_critical(n, alpha):
    """Two-sided Grubbs critical value for sample size n at significance alpha."""
    p = alpha / (2 * n)
    t = scipy_stats.t.ppf(1 - p, n - 2)
    g = ((n - 1) / np.sqrt(n)) * np.sqrt(t**2 / (n - 2 + t**2))
    return g


def sesd(series, max_anomalies_frac=0.10, alpha=0.05):
    """
    Seasonal (Generalised) ESD anomaly detector.

    Returns
    -------
    anomaly_dates : list of date strings
    scores        : pd.Series of Grubbs test statistics
    """
    x   = series.dropna().copy()
    n   = len(x)
    max_anom = max(1, int(np.floor(n * max_anomalies_frac)))

    residuals     = x - x.median()
    working       = residuals.copy()
    removed_idx   = []
    test_stats    = []
    critical_vals = []

    for i in range(max_anom):
        if len(working) < 3:
            break
        mean = working.mean()
        std  = working.std()
        if std < 1e-12:
            break
        R      = ((working - mean).abs() / std)
        idx    = R.idxmax()
        Rmax   = R[idx]
        gcrit  = _grubbs_critical(len(working), alpha)
        test_stats.append(float(Rmax))
        critical_vals.append(float(gcrit))
        removed_idx.append(idx)
        working = working.drop(idx)

    k = 0
    for j in range(len(test_stats) - 1, -1, -1):
        if test_stats[j] > critical_vals[j]:
            k = j + 1
            break

    anomaly_dates = removed_idx[:k]
    scores = pd.Series(
        test_stats,
        index=[str(d.date()) if hasattr(d, 'date') else str(d) for d in removed_idx],
        dtype=float,
    )
    return [str(d.date()) if hasattr(d, 'date') else str(d) for d in anomaly_dates], scores


def iqr_anomalies(series, k=1.5):
    """IQR fence anomaly detector."""
    x      = series.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask   = (x < lo) | (x > hi)
    return [str(d.date()) for d in x[mask].index], x[mask].to_dict()


def zscore_anomalies(series, threshold=3.0):
    """Z-score anomaly detector."""
    x    = series.dropna()
    z    = (x - x.mean()) / (x.std() + 1e-12)
    mask = z.abs() > threshold
    return ([str(d.date()) for d in x[mask].index],
            {str(d.date()): float(v) for d, v in z[mask].items()})


def rolling_sigma_anomalies(series, window=7, k=2.5):
    """Rolling σ anomaly detector."""
    x    = series.dropna()
    rm   = x.rolling(window, center=True, min_periods=3).mean()
    rs   = x.rolling(window, center=True, min_periods=3).std()
    dev  = (x - rm).abs()
    mask = (dev > k * rs).fillna(False)
    return ([str(d.date()) for d in x[mask].index],
            {str(d.date()): float(v) for d, v in dev[mask].items()})


def iforest_anomalies(series, contamination=0.05, random_state=42):
    """Isolation Forest anomaly detector."""
    x = series.dropna()
    if len(x) < 10:
        return [], {}
    X      = x.values.reshape(-1, 1)
    clf    = IsolationForest(contamination=contamination, random_state=random_state)
    preds  = clf.fit_predict(X)
    scores = clf.score_samples(X)
    mask   = preds == -1
    return ([str(d.date()) for d in x[mask].index],
            {str(d.date()): float(s) for d, s in zip(x[mask].index, scores[mask])})


def detect_anomalies_all(all_series, periods,
                          sesd_max_anomalies_frac, sesd_alpha,
                          iqr_k, zscore_thresh,
                          rolling_window, rolling_k,
                          iforest_contamination, iforest_random_state):
    """
    Run all five detectors for every run × every period.

    Returns
    -------
    dict  run_name -> {period_id -> {method -> result_dict}}
    """
    anomaly_results = {}

    for run_name, series in all_series.items():
        anomaly_results[run_name] = {}
        for i, (p_start, p_end) in enumerate(periods):
            mask = (series.index >= p_start) & (series.index <= p_end)
            slc  = series[mask].dropna()
            pid  = f'period_{i + 1}'

            if len(slc) < 5:
                anomaly_results[run_name][pid] = {'skipped': 'too few points'}
                continue

            sesd_dates, sesd_scores = sesd(slc, sesd_max_anomalies_frac, sesd_alpha)
            iqr_dates,  iqr_vals    = iqr_anomalies(slc, iqr_k)
            z_dates,    z_scores    = zscore_anomalies(slc, zscore_thresh)
            rs_dates,   rs_devs     = rolling_sigma_anomalies(slc, rolling_window, rolling_k)
            if_dates,   if_scores   = iforest_anomalies(slc, iforest_contamination, iforest_random_state)

            all_flagged = sesd_dates + iqr_dates + z_dates + rs_dates + if_dates
            vote_counts = Counter(all_flagged)
            consensus   = [d for d, cnt in vote_counts.items() if cnt >= 2]

            anomaly_results[run_name][pid] = {
                'period_start': str(p_start.date()),
                'period_end':   str(p_end.date()),
                'n_points':     int(len(slc)),
                'sesd': {
                    'anomaly_dates': sesd_dates,
                    'n_anomalies':   len(sesd_dates),
                    'scores':        sesd_scores.to_dict(),
                },
                'iqr': {
                    'anomaly_dates': iqr_dates,
                    'n_anomalies':   len(iqr_dates),
                    'values':        {str(k): float(v) for k, v in iqr_vals.items()},
                },
                'zscore': {
                    'anomaly_dates': z_dates,
                    'n_anomalies':   len(z_dates),
                    'z_scores':      {str(k): float(v) for k, v in z_scores.items()},
                },
                'rolling_sigma': {
                    'anomaly_dates': rs_dates,
                    'n_anomalies':   len(rs_dates),
                    'deviations':    {str(k): float(v) for k, v in rs_devs.items()},
                    'window':        rolling_window,
                    'k':             rolling_k,
                },
                'iforest': {
                    'anomaly_dates': if_dates,
                    'n_anomalies':   len(if_dates),
                    'scores':        {str(k): float(v) for k, v in if_scores.items()},
                    'contamination': iforest_contamination,
                },
                'consensus': {
                    'anomaly_dates':  sorted(consensus),
                    'n_anomalies':    len(consensus),
                    'vote_threshold': 2,
                    'votes':          dict(vote_counts),
                },
            }

    return anomaly_results


def print_anomaly_summary(anomaly_results):
    """Print a compact anomaly count table."""
    print('\nAnomaly counts (consensus ≥2 methods):')
    print(f'{"Run":<35} {"Period":<10} {"Days":<7} {"SESD":<6} {"IQR":<6} '
          f'{"Z":<6} {"RollΣ":<7} {"IFor":<6} {"Cons"}')
    print('-' * 95)
    for run_name in anomaly_results:
        for pid, pr in anomaly_results[run_name].items():
            if 'skipped' in pr:
                continue
            print(f'{run_name:<35} {pid:<10} {pr["n_points"]:<7} '
                  f'{pr["sesd"]["n_anomalies"]:<6} '
                  f'{pr["iqr"]["n_anomalies"]:<6} '
                  f'{pr["zscore"]["n_anomalies"]:<6} '
                  f'{pr["rolling_sigma"]["n_anomalies"]:<7} '
                  f'{pr["iforest"]["n_anomalies"]:<6} '
                  f'{pr["consensus"]["n_anomalies"]}')


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_analysis_results(output_file, config, all_series, stitch_log,
                           cp_dates, periods, period_stats, anomaly_results,
                           year_data, loaded_years):
    """
    Assemble and serialise all analysis results to a single JSON file.

    Output structure
    ----------------
    {
      "config": { ... },
      "change_points": [ date_str, ... ],
      "runs": {
        "<run_name>": {
          "distance_series":   { date_str: float, ... },
          "stitched_dates":    [ date_str, ... ],
          "pd_series":         { date_str: { dim: [[b,d],...], ... }, ... },
          "change_points":     [ date_str, ... ],
          "periods": [
            { "period_index": 1, "start": ..., "end": ...,
              "stats": {...}, "anomalies": {...} },
            ...
          ]
        }
      }
    }
    """
    output = {
        'config': {
            'year_start':              config['YEAR_START'],
            'year_end':                config['YEAR_END'],
            'runs':                    config['RUNS'],
            'cp_min':                  config['CP_MIN'],
            'cp_max':                  config['CP_MAX'],
            'cp_model':                config['CP_MODEL'],
            'sesd_max_anomalies_frac': config['SESD_MAX_ANOMALIES_FRAC'],
            'sesd_alpha':              config['SESD_ALPHA'],
            'iqr_k':                   config['IQR_K'],
            'zscore_thresh':           config['ZSCORE_THRESH'],
            'rolling_window':          config['ROLLING_WINDOW'],
            'rolling_k':               config['ROLLING_K'],
            'iforest_contamination':   config['IFOREST_CONTAMINATION'],
        },
        'change_points': [str(d.date()) for d in cp_dates],
        'runs': {},
    }

    for run_name in config['RUNS']:
        if run_name not in all_series:
            continue

        series = all_series[run_name]

        combined_pd = {}
        for year in loaded_years:
            if run_name not in year_data.get(year, {}):
                continue
            raw_pd = year_data[year][run_name].get('pd_series', {})
            combined_pd.update(raw_pd)

        periods_out = []
        for p in period_stats.get(run_name, []):
            pid = f'period_{p["period_index"]}'
            periods_out.append({
                'period_index': p['period_index'],
                'start':        p['start'],
                'end':          p['end'],
                'stats':        {k: v for k, v in p.items()
                                 if k not in ('period_index', 'start', 'end')},
                'anomalies':    anomaly_results.get(run_name, {}).get(pid, {}),
            })

        output['runs'][run_name] = {
            'distance_series': {str(d.date()): float(v) for d, v in series.items()},
            'stitched_dates':  stitch_log.get(run_name, []),
            'pd_series':       combined_pd,
            'change_points':   [str(d.date()) for d in cp_dates],
            'periods':         periods_out,
        }

    output = make_json_serializable(output)
    output = replace_nan(output)

    output_file = Path(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = output_file.stat().st_size / 1e6
    print(f'Saved → {output_file}  ({size_mb:.2f} MB)')
    print(f'  Runs saved   : {list(output["runs"].keys())}')
    print(f'  Change points: {output["change_points"]}')
    return output


print('period_analysis_functions.py loaded ✓')
