"""
period_visualisation_functions.py
===================================
Visualisation functions for multi-year period analysis results.
Loaded from the JSON produced by notebook 4b / period_analysis_functions.py.
"""

import warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_MARKERS = {
    'sesd':          ('D', 'red',     'S-ESD'),
    'iqr':           ('s', 'orange',  'IQR'),
    'zscore':        ('^', 'purple',  'Z-score'),
    'rolling_sigma': ('v', 'brown',   'Rolling σ'),
    'iforest':       ('P', 'magenta', 'Iso-Forest'),
    'consensus':     ('*', 'crimson', 'Consensus'),
}

STAT_COLS = [
    ('period_index', 'Period'),
    ('start',        'Start'),
    ('end',          'End'),
    ('n_days',       'N'),
    ('mean',         'Mean'),
    ('median',       'Median'),
    ('std',          'Std'),
    ('variance',     'Variance'),
    ('skewness',     'Skewness'),
    ('kurtosis',     'Kurtosis'),
    ('iqr',          'IQR'),
    ('cv',           'CV'),
    ('autocorr_lag1','AC-1'),
    ('autocorr_lag7','AC-7'),
    ('trend_slope',  'Trend'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_analysis(input_file):
    """Load the saved analysis JSON and return the dict."""
    with open(input_file, 'r') as f:
        analysis = json.load(f)
    return analysis


def parse_series(run_dict):
    """Reconstruct pd.Series from the stored distance_series dict."""
    ds = run_dict['distance_series']
    return pd.Series(
        list(ds.values()),
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in ds.keys()]),
        dtype=float,
    ).sort_index()


def parse_pd_series(run_dict):
    """Reconstruct {Timestamp: {int_dim: np.array}} from stored pd_series."""
    out = {}
    for date_str, dims in run_dict.get('pd_series', {}).items():
        out[pd.Timestamp(date_str)] = {int(k): np.array(v) for k, v in dims.items()}
    return out


def anomaly_dates_for_period(period_dict, methods):
    """Return {method: [Timestamp,...]} for the requested methods."""
    result = {}
    for method in methods:
        if method not in period_dict.get('anomalies', {}):
            continue
        raw = period_dict['anomalies'][method].get('anomaly_dates', [])
        result[method] = [pd.Timestamp(d) for d in raw]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_pd(ax, dgm_dict, title, color='steelblue', dims=None):
    """
    Draw a persistence diagram on ax, restricted to `dims`.
    dims=None shows all; dims=[1] shows H1 only.
    """
    if not dgm_dict:
        ax.set_visible(False)
        return

    all_pts = []
    styles  = {0: ('o', 0.55, 'H0'), 1: ('s', 0.80, 'H1')}
    keys    = sorted(int(k) for k in dgm_dict.keys())
    if dims is not None:
        keys = [k for k in keys if k in dims]

    for dim_key in keys:
        raw_key = str(dim_key) if str(dim_key) in dgm_dict else dim_key
        pts = np.array(dgm_dict[raw_key])
        if len(pts) == 0:
            continue
        finite = pts[pts[:, 1] < 1e9]
        if len(finite) == 0:
            continue
        marker, alpha, dlabel = styles.get(dim_key, ('o', 0.5, f'H{dim_key}'))
        ax.scatter(finite[:, 0], finite[:, 1],
                   marker=marker, s=30, alpha=alpha,
                   color=color, label=dlabel)
        all_pts.append(finite)

    if not all_pts:
        ax.text(0.5, 0.5, 'no pts', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='grey')
        ax.set_title(title, fontsize=7)
        return

    combined = np.vstack(all_pts)
    lo = combined.min() - 0.02
    hi = combined.max() + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.35)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Birth', fontsize=7)
    ax.set_ylabel('Death', fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='lower right')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_run(run_name, run_data, viz_methods, viz_pd_dims,
                  viz_normal_days, viz_fig_width, run_colors):
    """
    Produce all outputs for one run:
      - Period characterisation summary table (displayed inline)
      - Master anomaly table (displayed inline)
      - One figure per period (time-series + PD panels)

    Parameters
    ----------
    run_name       : str
    run_data       : dict  (one entry from analysis['runs'])
    viz_methods    : list of method keys, e.g. ['sesd', 'consensus']
    viz_pd_dims    : list of homology dims to show, e.g. [1]
    viz_normal_days: int   number of normal days to show PDs for
    viz_fig_width  : int/float figure width in inches
    run_colors     : dict  run_name -> colour string
    """
    from IPython.display import display   # only needed in notebook context

    series    = parse_series(run_data)
    pd_series = parse_pd_series(run_data)
    periods_  = run_data['periods']

    # ── Period characterisation table ────────────────────────────────────────
    stat_rows = []
    for p in periods_:
        s = p.get('stats', {})
        row = {'Period': p['period_index'], 'Start': p['start'], 'End': p['end']}
        for key, col in STAT_COLS[3:]:
            v = s.get(key, float('nan'))
            row[col] = round(v, 4) if isinstance(v, float) else v
        stat_rows.append(row)
    stat_df = pd.DataFrame(stat_rows)

    print(f'\n{"="*80}')
    print(f'  RUN: {run_name}')
    print(f'{"="*80}')
    print('\n── Period characterisation ──')
    display(stat_df.style
        .format({c: '{:.4f}' for c in stat_df.select_dtypes('float').columns})
        .set_caption(f'{run_name} — period statistics')
        .set_table_styles([{'selector': 'th', 'props': [('font-size', '11px')]}])
    )

    # ── Master anomaly table ─────────────────────────────────────────────────
    anom_rows = []
    for p in periods_:
        p_idx      = p['period_index']
        p_start    = pd.Timestamp(p['start'])
        p_end      = pd.Timestamp(p['end'])
        anom_block = p.get('anomalies', {})
        date_method_map = {}
        for method in viz_methods:
            if method not in anom_block:
                continue
            for d in anom_block[method].get('anomaly_dates', []):
                date_method_map.setdefault(d, set()).add(method)
        for date_str, methods_set in sorted(date_method_map.items()):
            ts  = pd.Timestamp(date_str)
            val = series.get(ts, float('nan'))
            anom_rows.append({
                'Run':       run_name,
                'Period':    p_idx,
                'P-start':   str(p_start.date()),
                'P-end':     str(p_end.date()),
                'Date':      date_str,
                'W1':        round(float(val), 4) if not np.isnan(val) else float('nan'),
                'Methods':   ', '.join(sorted(methods_set)),
                'N-methods': len(methods_set),
            })

    if anom_rows:
        anom_df = pd.DataFrame(anom_rows).sort_values(['Period', 'Date'])
        print('\n── Anomaly table ──')
        display(anom_df.style
            .background_gradient(subset=['N-methods'], cmap='Reds', vmin=1)
            .format({'W1': '{:.4f}'})
            .set_caption(f'{run_name} — all anomalies ({len(anom_df)} total)')
            .set_table_styles([{'selector': 'th', 'props': [('font-size', '11px')]}])
        )
    else:
        print('  (no anomalies found for selected methods)')

    # ── Per-period figures ───────────────────────────────────────────────────
    for period in periods_:
        p_idx   = period['period_index']
        p_start = pd.Timestamp(period['start'])
        p_end   = pd.Timestamp(period['end'])
        stats   = period.get('stats', {})
        n_days  = stats.get('n_days', 0)

        mask = (series.index >= p_start) & (series.index <= p_end)
        slc  = series[mask]
        if len(slc) == 0:
            continue

        method_anom  = anomaly_dates_for_period(period, viz_methods)
        all_anom_ts  = sorted(set(t for ts in method_anom.values() for t in ts))

        anom_set     = set(str(t.date()) for t in all_anom_ts)
        candidates   = [t for t in slc.index
                        if str(t.date()) not in anom_set and t in pd_series]
        med          = slc.median()
        candidates   = sorted(candidates, key=lambda t: abs(slc[t] - med))
        normal_days  = candidates[:viz_normal_days]

        anom_days_with_pd = [t for t in all_anom_ts if t in pd_series][:3]
        pd_days           = anom_days_with_pd + normal_days
        n_pd_cols         = len(pd_days)

        n_cols = max(n_pd_cols, 1)
        fig_h  = 4.5 + 3.2 * (1 if n_pd_cols > 0 else 0)
        fig    = plt.figure(figsize=(viz_fig_width, fig_h))
        gs     = gridspec.GridSpec(
            2 if n_pd_cols > 0 else 1, n_cols,
            height_ratios=[2.5, 1] if n_pd_cols > 0 else [1],
            hspace=0.45, wspace=0.35,
        )

        # Time-series panel
        ax_ts = fig.add_subplot(gs[0, :])
        ax_ts.plot(slc.index, slc.values,
                   color=run_colors[run_name], lw=1.2, alpha=0.85)

        rm = slc.rolling(7, center=True, min_periods=3).mean()
        rs = slc.rolling(7, center=True, min_periods=3).std()
        ax_ts.fill_between(slc.index, rm - rs, rm + rs,
                           alpha=0.15, color=run_colors[run_name])
        ax_ts.plot(rm.index, rm.values, '--',
                   lw=0.8, color=run_colors[run_name], alpha=0.6)

        legend_handles = [Line2D([0], [0], color=run_colors[run_name],
                                  lw=1.5, label=run_name)]
        for method, anom_ts in method_anom.items():
            marker, color, label = METHOD_MARKERS.get(method, ('o', 'grey', method))
            anom_in_slc = [t for t in anom_ts if t in slc.index]
            if anom_in_slc:
                ax_ts.scatter(anom_in_slc, slc[anom_in_slc].values,
                              marker=marker, s=70, color=color,
                              zorder=5, edgecolors='k', linewidths=0.4)
            legend_handles.append(
                Line2D([0], [0], marker=marker, color='w',
                       markerfacecolor=color, markersize=8,
                       markeredgecolor='k',
                       label=f'{label} ({len(anom_in_slc)})')
            )
        if normal_days:
            ax_ts.scatter(normal_days, slc[normal_days].values,
                          marker='o', s=40, color='limegreen', zorder=4,
                          edgecolors='darkgreen', linewidths=0.5)
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='limegreen', markersize=7,
                       markeredgecolor='darkgreen', label='Normal (PD shown)')
            )

        ax_ts.set_xlim(p_start, p_end)
        ax_ts.set_ylabel('W₁ distance', fontsize=9)
        ax_ts.tick_params(axis='x', labelrotation=30, labelsize=8)
        stat_txt = (
            f"n={n_days}  "
            f"μ={stats.get('mean', float('nan')):.3f}  "
            f"med={stats.get('median', float('nan')):.3f}  "
            f"σ={stats.get('std', float('nan')):.3f}  "
            f"skew={stats.get('skewness', float('nan')):.2f}  "
            f"kurt={stats.get('kurtosis', float('nan')):.2f}  "
            f"trend={stats.get('trend_slope', float('nan')):.2e}"
        )
        ax_ts.set_title(
            f'Run: {run_name}  |  Period {p_idx}: '
            f'{p_start.date()} → {p_end.date()}\n{stat_txt}',
            fontsize=9, pad=6,
        )
        ax_ts.legend(handles=legend_handles, fontsize=7,
                     loc='upper right', framealpha=0.7)

        # PD panels
        if n_pd_cols > 0:
            for col_i, day_ts in enumerate(pd_days):
                ax_pd   = fig.add_subplot(gs[1, col_i])
                dgm     = pd_series.get(day_ts, {})
                is_anom = day_ts in anom_days_with_pd
                pd_color = 'firebrick' if is_anom else 'seagreen'
                prefix   = '⚠ ANOM' if is_anom else '✓ normal'
                flagged  = [m for m, ts_list in method_anom.items()
                            if day_ts in ts_list]
                subtitle = ('\n' + ', '.join(flagged)) if flagged else ''
                _plot_pd(ax_pd, dgm,
                         title=f'{prefix}{subtitle}\n{day_ts.date()}',
                         color=pd_color, dims=viz_pd_dims)

        plt.tight_layout()
        plt.show()
        print()


def visualise_all(analysis, viz_methods, viz_pd_dims,
                  viz_normal_days, viz_fig_width):
    """
    Run visualise_run for every run in the analysis dict.

    Parameters
    ----------
    analysis       : dict  loaded from the analysis JSON
    viz_methods    : list of method keys
    viz_pd_dims    : list of homology dims
    viz_normal_days: int
    viz_fig_width  : int/float
    """
    viz_runs  = list(analysis['runs'].keys())
    _palette  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    run_colors = {r: _palette[i % len(_palette)] for i, r in enumerate(viz_runs)}

    viz_cp = [pd.Timestamp(d) for d in analysis['change_points']]
    print(f'Runs    : {viz_runs}')
    print(f'CP dates: {[str(d.date()) for d in viz_cp]}')
    print(f'Methods : {viz_methods}')

    for run_name in viz_runs:
        visualise_run(
            run_name      = run_name,
            run_data      = analysis['runs'][run_name],
            viz_methods   = viz_methods,
            viz_pd_dims   = viz_pd_dims,
            viz_normal_days = viz_normal_days,
            viz_fig_width = viz_fig_width,
            run_colors    = run_colors,
        )


print('period_visualisation_functions.py loaded ✓')
