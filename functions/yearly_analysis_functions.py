"""
Yearly Anomaly Analysis Functions
==================================
Multi-method anomaly detection for Wasserstein distance series.

Methods implemented:
  1. S-ESD  (Seasonal Extreme Studentised Deviate)
  2. IQR    (Interquartile Range fence)
  3. Z-Score (3-sigma rule on rolling-window residuals)
  4. Isolation Forest
  5. LOF    (Local Outlier Factor)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION METHODS
# ═══════════════════════════════════════════════════════════════════════════

def generalized_esd(series, max_anomalies=None, alpha=0.05):
    """Rosner's Generalised ESD test. Returns a boolean anomaly mask."""
    x = series.dropna().values.copy()
    n = len(x)
    if max_anomalies is None:
        max_anomalies = max(1, int(0.10 * n))
    anomaly_idx = []
    remaining   = list(range(n))
    for k in range(1, max_anomalies + 1):
        if len(remaining) < 3:
            break
        sub       = x[remaining]
        mu, sigma = sub.mean(), sub.std(ddof=1)
        if sigma < 1e-10:
            break
        dev       = np.abs(sub - mu) / sigma
        local_idx = int(np.argmax(dev))
        R_k       = dev[local_idx]
        p         = alpha / (2 * (n - k + 1))
        df_       = n - k - 1
        t_k       = stats.t.ppf(1 - p, df=df_) if df_ > 0 else 0.0
        lam       = ((n - k) * t_k) / np.sqrt((df_ + t_k**2) * (n - k + 1))
        if R_k > lam:
            anomaly_idx.append(remaining[local_idx])
            remaining.pop(local_idx)
        else:
            break
    mask = np.zeros(n, dtype=bool)
    mask[anomaly_idx] = True
    return mask


def s_esd(series, period=7, alpha=0.05, max_anomalies=None):
    """S-ESD: detrend + deseasonalise with rolling-median, then apply GESD."""
    s        = series.copy().ffill().fillna(0)
    trend    = s.rolling(window=period * 2 + 1, center=True, min_periods=1).median()
    detrend  = s - trend
    seasonal = detrend.groupby(detrend.index.dayofweek).transform('median')
    residual = detrend - seasonal
    clean    = residual.dropna()
    mask     = generalized_esd(clean, max_anomalies=max_anomalies, alpha=alpha)
    anomalies = pd.Series(False, index=series.index)
    anomalies.loc[clean.index[mask]] = True
    return anomalies, residual


def detect_iqr(series, k=1.5):
    """IQR fence method. Flags values below Q1 - k*IQR or above Q3 + k*IQR."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi), lo, hi


def detect_zscore(series, threshold=3.0, window=30):
    """Rolling Z-score with centered window."""
    roll_mean  = series.rolling(window, center=True, min_periods=5).mean()
    roll_std   = series.rolling(window, center=True, min_periods=5).std(ddof=1)
    z          = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.abs() > threshold, z


def detect_isolation_forest(series, contamination=0.05, random_state=42):
    """Isolation Forest on the raw values (univariate)."""
    X   = series.values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(X)
    scores = -clf.score_samples(X)
    return pd.Series(preds == -1, index=series.index), pd.Series(scores, index=series.index)


def detect_lof(series, n_neighbors=20, contamination=0.05):
    """Local Outlier Factor on the raw values (univariate)."""
    X   = series.values.reshape(-1, 1)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds  = clf.fit_predict(X)
    scores = -clf.negative_outlier_factor_
    return pd.Series(preds == -1, index=series.index), pd.Series(scores, index=series.index)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def statistical_summary(series):
    """Return a dict of descriptive + distributional statistics."""
    q1, q3   = series.quantile(0.25), series.quantile(0.75)
    sk_stat, sk_p   = stats.skewtest(series.dropna())
    ku_stat, ku_p   = stats.kurtosistest(series.dropna())
    sh_stat, sh_p   = stats.shapiro(series.dropna()[:min(5000, len(series))])
    autocorr_1      = series.autocorr(lag=1)
    autocorr_7      = series.autocorr(lag=7)

    return {
        "N"                    : len(series),
        "Mean"                 : series.mean(),
        "Median"               : series.median(),
        "Std Dev"              : series.std(ddof=1),
        "Min"                  : series.min(),
        "Max"                  : series.max(),
        "Range"                : series.max() - series.min(),
        "IQR"                  : q3 - q1,
        "Q1"                   : q1,
        "Q3"                   : q3,
        "Skewness"             : series.skew(),
        "Skewness p-value"     : sk_p,
        "Excess Kurtosis"      : series.kurtosis(),
        "Kurtosis p-value"     : ku_p,
        "Shapiro-Wilk stat"    : sh_stat,
        "Shapiro-Wilk p-value" : sh_p,
        "Autocorr (lag-1)"     : autocorr_1,
        "Autocorr (lag-7)"     : autocorr_7,
        "CV (%)"               : 100 * series.std() / series.mean(),
    }


def print_summary(summary, method_counts):
    """Print statistical summary and anomaly counts."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  WASSERSTEIN DISTANCE — STATISTICAL OVERVIEW")
    print(sep)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:>12.4f}")
        else:
            print(f"  {k:<28} {v:>12}")
    print(f"\n{sep}")
    print("  ANOMALY DETECTION SUMMARY")
    print(sep)
    total = summary["N"]
    for method, count in method_counts.items():
        pct = 100 * count / total
        print(f"  {method:<28} {count:>4} flagged  ({pct:.1f}%)")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE VOTE
# ═══════════════════════════════════════════════════════════════════════════

def ensemble_vote(flags_dict, min_votes=3):
    """Flag a point if at least `min_votes` methods agree."""
    df     = pd.DataFrame(flags_dict).astype(int)
    votes  = df.sum(axis=1)
    return votes >= min_votes, votes


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg"        : "#0d0f14",
    "panel"     : "#13161e",
    "grid"      : "#1e2130",
    "text"      : "#c8cfe0",
    "muted"     : "#5a6280",
    "accent"    : "#7eb8f7",
    "series"    : "#4a9eda",
    "sesd"      : "#f0a653",
    "iqr"       : "#a78bfa",
    "zscore"    : "#34d399",
    "iforest"   : "#f87171",
    "lof"       : "#fb923c",
    "ensemble"  : "#f43f5e",
}

METHOD_COLORS = {
    "S-ESD"             : PALETTE["sesd"],
    "IQR"               : PALETTE["iqr"],
    "Z-Score"           : PALETTE["zscore"],
    "Isolation Forest"  : PALETTE["iforest"],
    "LOF"               : PALETTE["lof"],
    "Ensemble"          : PALETTE["ensemble"],
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply dark theme styling to axis."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=9, fontweight="bold",
                     pad=6, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def plot_analysis(series, flags, scores, residuals, iqr_bounds, z_scores,
                  ensemble_mask, vote_counts, summary, method_counts,
                  title_suffix="", output_path="wasserstein_anomaly_report.png"):
    """Create comprehensive anomaly analysis visualization."""

    plt.rcParams.update({
        "font.family"    : "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color"     : PALETTE["text"],
    })

    fig = plt.figure(figsize=(20, 22), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(
        5, 3,
        figure=fig,
        hspace=0.52, wspace=0.35,
        top=0.93, bottom=0.05, left=0.07, right=0.97
    )

    # Title
    title_text = f"WASSERSTEIN DISTANCE  ·  ANOMALY ANALYSIS{title_suffix}"
    fig.text(0.5, 0.965, title_text,
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=PALETTE["text"], fontfamily="monospace",
             bbox=dict(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
                       boxstyle="round,pad=0.5", linewidth=1))

    # Stat bar
    stat_ax = fig.add_subplot(gs[0, :])
    stat_ax.set_visible(False)
    stats_labels = [
        ("N", f"{summary['N']}"),
        ("Mean", f"{summary['Mean']:.2f}"),
        ("Std", f"{summary['Std Dev']:.2f}"),
        ("Skew", f"{summary['Skewness']:.2f}"),
        ("Kurt", f"{summary['Excess Kurtosis']:.2f}"),
        ("CV", f"{summary['CV (%)']:.1f}%"),
        ("AC1", f"{summary['Autocorr (lag-1)']:.3f}"),
        ("AC7", f"{summary['Autocorr (lag-7)']:.3f}"),
        ("Shapiro p", f"{summary['Shapiro-Wilk p-value']:.3f}"),
        ("IQR", f"{summary['IQR']:.2f}"),
    ]
    for method, cnt in method_counts.items():
        stats_labels.append((method, f"{cnt} anomalies"))

    n_chips  = len(stats_labels)
    chip_w   = 1.0 / n_chips
    chip_ax  = fig.add_axes([0.03, 0.915, 0.94, 0.035])
    chip_ax.set_facecolor(PALETTE["bg"])
    chip_ax.set_xlim(0, 1); chip_ax.set_ylim(0, 1)
    chip_ax.axis("off")
    for i, (lbl, val) in enumerate(stats_labels):
        x  = i * chip_w + chip_w * 0.1
        color = METHOD_COLORS.get(lbl, PALETTE["accent"])
        chip_ax.text(x, 0.72, lbl, color=PALETTE["muted"],
                     fontsize=6.5, fontweight="bold", va="center")
        chip_ax.text(x, 0.25, val, color=color,
                     fontsize=7.5, fontweight="bold", va="center")

    # Main series plot
    ax_main = fig.add_subplot(gs[1, :])
    _style_ax(ax_main, title="Raw Series · All Anomaly Methods", ylabel="Wasserstein Distance")
    ax_main.plot(series.index, series.values,
                 color=PALETTE["series"], lw=0.9, alpha=0.85, zorder=2, label="Distance")
    lo, hi = iqr_bounds
    ax_main.axhline(hi, color=PALETTE["iqr"], lw=0.8, ls="--", alpha=0.6, label=f"IQR upper {hi:.2f}")
    ax_main.axhline(lo, color=PALETTE["iqr"], lw=0.8, ls=":",  alpha=0.6, label=f"IQR lower {lo:.2f}")

    scatter_offsets = np.linspace(-0.3, 0.3, 5)
    for offset, (method, flag) in zip(scatter_offsets, flags.items()):
        color = METHOD_COLORS[method]
        idx   = series[flag].index
        if len(idx):
            ax_main.scatter(idx, series[flag] + offset,
                            color=color, s=22, alpha=0.75, zorder=5, label=method, marker="v")

    for dt in series[ensemble_mask].index:
        ax_main.axvspan(dt - pd.Timedelta("0.5D"), dt + pd.Timedelta("0.5D"),
                        color=PALETTE["ensemble"], alpha=0.15)

    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_main.legend(loc="upper right", fontsize=6.5, ncol=5,
                   facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
                   labelcolor=PALETTE["text"])

    # Residuals
    ax_resid = fig.add_subplot(gs[2, 0])
    _style_ax(ax_resid, title="S-ESD Residuals", ylabel="Residual")
    ax_resid.plot(residuals.index, residuals.values,
                  color=PALETTE["accent"], lw=0.8, alpha=0.7)
    sesd_idx = series[flags["S-ESD"]].index
    ax_resid.scatter(sesd_idx, residuals.loc[sesd_idx],
                     color=PALETTE["sesd"], s=35, zorder=5, marker="^")
    ax_resid.axhline(0, color=PALETTE["muted"], lw=0.6, ls="--")
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Z-score
    ax_z = fig.add_subplot(gs[2, 1])
    _style_ax(ax_z, title="Rolling Z-Score (window=30)", ylabel="|Z|")
    ax_z.fill_between(z_scores.index, z_scores.abs(),
                      color=PALETTE["zscore"], alpha=0.25)
    ax_z.plot(z_scores.index, z_scores.abs(),
              color=PALETTE["zscore"], lw=0.8, alpha=0.9)
    ax_z.axhline(3, color=PALETTE["muted"], lw=0.8, ls="--", label="|Z|=3")
    z_idx = series[flags["Z-Score"]].index
    ax_z.scatter(z_idx, z_scores.loc[z_idx].abs(),
                 color=PALETTE["series"], s=30, zorder=5, marker="D")
    ax_z.legend(fontsize=7, facecolor=PALETTE["panel"],
                edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Vote heatmap
    ax_votes = fig.add_subplot(gs[2, 2])
    _style_ax(ax_votes, title="Ensemble Vote Count", ylabel="# Methods Agree")
    cmap = plt.cm.get_cmap("YlOrRd")
    colors = [cmap(v / 5) for v in vote_counts]
    ax_votes.bar(vote_counts.index, vote_counts.values,
                 color=colors, width=1.0, edgecolor="none", alpha=0.9)
    ax_votes.axhline(3, color=PALETTE["ensemble"], lw=1.0, ls="--", label="Ensemble threshold")
    ax_votes.set_ylim(0, 5.5)
    ax_votes.set_yticks(range(6))
    ax_votes.legend(fontsize=7, facecolor=PALETTE["panel"],
                    edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax_votes.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Isolation Forest
    ax_if = fig.add_subplot(gs[3, 0])
    _style_ax(ax_if, title="Isolation Forest Score", ylabel="Anomaly Score")
    sc_if = scores["Isolation Forest"]
    ax_if.fill_between(sc_if.index, sc_if,
                       color=PALETTE["iforest"], alpha=0.2)
    ax_if.plot(sc_if.index, sc_if, color=PALETTE["iforest"], lw=0.8)
    if_idx = series[flags["Isolation Forest"]].index
    ax_if.scatter(if_idx, sc_if.loc[if_idx],
                  color=PALETTE["ensemble"], s=28, zorder=5, marker="*")
    ax_if.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # LOF
    ax_lof = fig.add_subplot(gs[3, 1])
    _style_ax(ax_lof, title="LOF Score", ylabel="LOF Score")
    sc_lof = scores["LOF"]
    ax_lof.fill_between(sc_lof.index, sc_lof,
                        color=PALETTE["lof"], alpha=0.2)
    ax_lof.plot(sc_lof.index, sc_lof, color=PALETTE["lof"], lw=0.8)
    lof_idx = series[flags["LOF"]].index
    ax_lof.scatter(lof_idx, sc_lof.loc[lof_idx],
                   color=PALETTE["ensemble"], s=28, zorder=5, marker="*")
    ax_lof.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Distribution
    ax_dist = fig.add_subplot(gs[3, 2])
    _style_ax(ax_dist, title="Value Distribution", xlabel="Distance", ylabel="Density")
    vals = series.dropna().values
    ax_dist.hist(vals, bins=40, color=PALETTE["series"],
                 alpha=0.35, density=True, edgecolor="none")
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    kde   = stats.gaussian_kde(vals)
    ax_dist.plot(kde_x, kde(kde_x), color=PALETTE["accent"], lw=1.5)
    lo_b, hi_b = iqr_bounds
    ax_dist.axvline(hi_b, color=PALETTE["iqr"], lw=1.0, ls="--",
                    label=f"IQR hi {hi_b:.1f}")
    ax_dist.axvline(lo_b, color=PALETTE["iqr"], lw=1.0, ls=":",
                    label=f"IQR lo {lo_b:.1f}")
    mu, sigma = vals.mean(), vals.std()
    for nsig, alpha_ in [(2, 0.12), (3, 0.07)]:
        ax_dist.axvspan(mu - nsig * sigma, mu + nsig * sigma,
                        color=PALETTE["series"], alpha=alpha_)
    ax_dist.legend(fontsize=7, facecolor=PALETTE["panel"],
                   edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    # Method comparison heatmap
    ax_heat = fig.add_subplot(gs[4, :])
    _style_ax(ax_heat, title="Method Agreement · Daily Heatmap (1 = flagged anomaly)")
    method_names = list(flags.keys()) + ["Ensemble"]
    all_flags    = {**flags, "Ensemble": ensemble_mask}
    heat_data    = np.array([all_flags[m].astype(int).values for m in method_names])
    im = ax_heat.imshow(heat_data, aspect="auto", cmap="YlOrRd",
                        interpolation="none", vmin=0, vmax=1,
                        extent=[0, len(series), len(method_names), 0])
    ax_heat.set_yticks(np.arange(len(method_names)) + 0.5)
    ax_heat.set_yticklabels(method_names, fontsize=8, color=PALETTE["text"])
    month_ticks = series.resample("MS").first().index
    x_positions = [(series.index.get_loc(dt) if dt in series.index else
                    np.searchsorted(series.index, dt)) for dt in month_ticks]
    ax_heat.set_xticks(x_positions)
    ax_heat.set_xticklabels([dt.strftime("%b") for dt in month_ticks],
                             fontsize=7, color=PALETTE["muted"])
    ax_heat.tick_params(axis="y", length=0)

    plt.colorbar(im, ax=ax_heat, fraction=0.008, pad=0.01,
                 label="Anomaly flag").ax.yaxis.set_tick_params(color=PALETTE["muted"])

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"[✓] Figure saved → {output_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(distance_series: pd.Series,
                      sesd_period      : int   = 7,
                      sesd_alpha       : float = 0.05,
                      iqr_k            : float = 1.5,
                      zscore_threshold : float = 3.0,
                      zscore_window    : int   = 30,
                      contamination    : float = 0.05,
                      lof_neighbors    : int   = 20,
                      ensemble_votes   : int   = 3,
                      title_suffix     : str   = "",
                      output_path      : str   = "wasserstein_anomaly_report.png",
                      verbose          : bool  = True):
    """
    End-to-end anomaly analysis pipeline.

    Parameters
    ----------
    distance_series   : pd.Series with DatetimeIndex
    sesd_period       : seasonal period (days) for S-ESD
    sesd_alpha        : significance level for GESD
    iqr_k             : IQR fence multiplier
    zscore_threshold  : |Z| cutoff for rolling Z-score
    zscore_window     : rolling window for Z-score (days)
    contamination     : expected anomaly fraction for IF and LOF
    lof_neighbors     : n_neighbors for LOF
    ensemble_votes    : minimum method agreement
    title_suffix      : additional text for plot title
    output_path       : where to save the figure
    verbose           : print summary and anomaly dates
    """
    if verbose:
        print("Running anomaly detection…")

    # Run all methods
    sesd_mask, residuals = s_esd(distance_series, period=sesd_period, alpha=sesd_alpha)
    iqr_mask, lo_b, hi_b = detect_iqr(distance_series, k=iqr_k)
    z_mask, z_scores = detect_zscore(distance_series,
                                     threshold=zscore_threshold,
                                     window=zscore_window)
    if_mask, if_scores = detect_isolation_forest(distance_series,
                                                 contamination=contamination)
    lof_mask, lof_scores = detect_lof(distance_series,
                                      n_neighbors=lof_neighbors,
                                      contamination=contamination)

    flags = {
        "S-ESD"            : sesd_mask,
        "IQR"              : iqr_mask,
        "Z-Score"          : z_mask,
        "Isolation Forest" : if_mask,
        "LOF"              : lof_mask,
    }
    ensemble_mask, vote_counts = ensemble_vote(flags, min_votes=ensemble_votes)

    scores = {
        "Isolation Forest" : if_scores,
        "LOF"              : lof_scores,
    }

    # Statistics
    summary = statistical_summary(distance_series)
    method_counts = {m: int(f.sum()) for m, f in flags.items()}
    method_counts["Ensemble"] = int(ensemble_mask.sum())

    if verbose:
        print_summary(summary, method_counts)
        for method, flag in {**flags, "Ensemble": ensemble_mask}.items():
            dates = distance_series.index[flag]
            print(f"{method} flagged {len(dates)} days:")
            for d in sorted(dates):
                print(f"    {d.date()}  val={distance_series.loc[d]:.4f}")
            print()

    # Visualize
    plot_analysis(
        series         = distance_series,
        flags          = flags,
        scores         = scores,
        residuals      = residuals,
        iqr_bounds     = (lo_b, hi_b),
        z_scores       = z_scores,
        ensemble_mask  = ensemble_mask,
        vote_counts    = vote_counts,
        summary        = summary,
        method_counts  = method_counts,
        title_suffix   = title_suffix,
        output_path    = output_path,
    )

    return {
        "flags"        : flags,
        "ensemble"     : ensemble_mask,
        "votes"        : vote_counts,
        "residuals"    : residuals,
        "z_scores"     : z_scores,
        "if_scores"    : if_scores,
        "lof_scores"   : lof_scores,
        "summary"      : summary,
        "method_counts": method_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-YEAR & MULTI-RUN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_df(run_data_dict, all_results):
    """
    Build a tidy comparison DataFrame from already-computed results.

    Parameters
    ----------
    run_data_dict : dict  {(year, layer): pd.Series}
    all_results   : dict  {(year, layer): result_dict}  — output of run_full_analysis

    Returns
    -------
    pd.DataFrame  one row per (year, layer) run
    """
    rows = []
    for label, result in all_results.items():
        label_str = f"{label[0]}_{label[1]}" if isinstance(label, tuple) else str(label)
        row = {"label": label_str}
        row.update(result["summary"])
        row.update({f"{m}_count": c for m, c in result["method_counts"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(run_data_dict, all_results, output_dir="./anomaly_reports",
                    output_path=None):
    """
    Two-panel comparison figure:
      Top    — styled summary table (stats + anomaly counts per run)
      Bottom — unified anomaly timeline: all runs stacked, every ensemble
               anomaly marked with its characterization

    Parameters
    ----------
    run_data_dict : dict  {(year, layer): pd.Series}
    all_results   : dict  {(year, layer): result_dict}
    output_dir    : str   directory for saving (created if absent)
    output_path   : str   full path override; defaults to output_dir/comparison.png
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(output_dir, "comparison.png")

    labels   = list(all_results.keys())
    n_runs   = len(labels)

    plt.rcParams.update({
        "font.family"     : "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color"      : PALETTE["text"],
    })

    # ── layout: table row + one timeline row per run ─────────────────────────
    fig_height = 4 + n_runs * 2.2
    fig = plt.figure(figsize=(22, fig_height), facecolor=PALETTE["bg"])

    gs = plt.GridSpec(
        n_runs + 1, 1,
        figure=fig,
        hspace=0.55,
        top=0.93, bottom=0.05, left=0.18, right=0.97,
        height_ratios=[max(2.5, n_runs * 0.55)] + [1.0] * n_runs,
    )

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965,
             "WASSERSTEIN DISTANCE  ·  MULTI-RUN COMPARISON",
             ha="center", va="center", fontsize=15, fontweight="bold",
             color=PALETTE["text"], fontfamily="monospace",
             bbox=dict(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
                       boxstyle="round,pad=0.5", linewidth=1))

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 1 — Summary table
    # ══════════════════════════════════════════════════════════════════════════
    ax_tbl = fig.add_subplot(gs[0])
    ax_tbl.set_facecolor(PALETTE["panel"])
    for spine in ax_tbl.spines.values():
        spine.set_visible(False)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    ax_tbl.set_title("Run Statistics & Anomaly Counts",
                     color=PALETTE["text"], fontsize=9, fontweight="bold",
                     loc="left", pad=6)

    stat_cols   = ["N", "Mean", "Std Dev", "Min", "Max", "Skewness", "CV (%)"]
    method_cols = ["S-ESD_count", "IQR_count", "Z-Score_count",
                   "Isolation Forest_count", "LOF_count", "Ensemble_count"]
    col_labels  = ["Run", "N", "Mean", "Std", "Min", "Max", "Skew", "CV%",
                   "S-ESD", "IQR", "Z-Score", "IF", "LOF", "Ensemble"]

    df = build_comparison_df(run_data_dict, all_results)

    table_data = []
    for _, row in df.iterrows():
        r = [row["label"]]
        r += [f"{row[c]:.0f}" if c == "N" else f"{row[c]:.2f}" for c in stat_cols]
        r += [str(int(row[c])) for c in method_cols]
        table_data.append(r)

    n_cols = len(col_labels)
    col_w  = 1.0 / n_cols
    row_h  = 1.0 / (n_runs + 1.5)

    # Header row
    for j, lbl in enumerate(col_labels):
        x = (j + 0.5) * col_w
        ax_tbl.text(x, 1 - row_h * 0.5, lbl,
                    ha="center", va="center",
                    color=PALETTE["accent"], fontsize=7.5, fontweight="bold",
                    transform=ax_tbl.transAxes)
        # vertical divider
        if j > 0:
            ax_tbl.axvline(j * col_w, color=PALETTE["grid"], lw=0.5)

    ax_tbl.axhline(1 - row_h, color=PALETTE["grid"], lw=0.8)

    method_col_start = len(stat_cols) + 1   # index of first method column
    for i, row_vals in enumerate(table_data):
        y = 1 - row_h * (i + 1.5)
        bg_color = PALETTE["panel"] if i % 2 == 0 else "#161929"
        ax_tbl.axhspan(y - row_h * 0.5, y + row_h * 0.5,
                       facecolor=bg_color)
        for j, val in enumerate(row_vals):
            x = (j + 0.5) * col_w
            # colour method counts by magnitude
            if j >= method_col_start:
                method_name = col_labels[j]
                base_color  = METHOD_COLORS.get(
                    method_name.replace("_count", "").replace("IF", "Isolation Forest"),
                    PALETTE["text"]
                )
                txt_color = base_color if int(val) > 0 else PALETTE["muted"]
            else:
                txt_color = PALETTE["text"]
            ax_tbl.text(x, y, val,
                        ha="center", va="center",
                        color=txt_color, fontsize=7,
                        transform=ax_tbl.transAxes)

    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 2…N+1 — one timeline strip per run
    # ══════════════════════════════════════════════════════════════════════════
    run_colors = [
        "#4a9eda", "#f0a653", "#a78bfa", "#34d399",
        "#f87171", "#fb923c", "#7eb8f7", "#f43f5e",
    ]

    for idx, label in enumerate(labels):
        series = run_data_dict[label]
        result = all_results[label]
        label_str = f"{label[0]}  ·  {label[1]}" if isinstance(label, tuple) else str(label)
        run_color = run_colors[idx % len(run_colors)]

        ax = fig.add_subplot(gs[idx + 1])
        _style_ax(ax, ylabel="W-dist")
        ax.set_title(label_str, color=run_color, fontsize=8,
                     fontweight="bold", loc="left", pad=4)

        # Distance series
        ax.plot(series.index, series.values,
                color=run_color, lw=0.85, alpha=0.7, zorder=2)

        # Ensemble anomaly shading + characterization labels
        ensemble_dates = series.index[result["ensemble"]]
        flags          = result["flags"]
        votes          = result["votes"]

        for dt in ensemble_dates:
            ax.axvspan(dt - pd.Timedelta("0.5D"), dt + pd.Timedelta("0.5D"),
                       color=PALETTE["ensemble"], alpha=0.25, zorder=3)

            # which methods fired?
            methods_fired = [m for m, flag in flags.items() if dt in flag.index and flag.loc[dt]]
            n_votes = int(votes.loc[dt] if dt in votes.index else 0)
            val     = series.loc[dt] if dt in series.index else float("nan")

            # short label: vote count + value
            lbl = f"{n_votes}v\n{val:.1f}"
            ax.text(dt, series.max() * 0.92, lbl,
                    ha="center", va="top", fontsize=5.5,
                    color=PALETTE["ensemble"], fontweight="bold", zorder=5)

        # Non-ensemble per-method markers (small, semi-transparent)
        method_marker = {"S-ESD": "^", "IQR": "D", "Z-Score": "s",
                         "Isolation Forest": "*", "LOF": "P"}
        scatter_y_offsets = np.linspace(0.08, 0.32, 5)
        for offset, (method, flag) in zip(scatter_y_offsets, flags.items()):
            flagged = series[flag]
            if len(flagged):
                ax.scatter(flagged.index,
                           [series.min() + offset * (series.max() - series.min())] * len(flagged),
                           color=METHOD_COLORS[method], s=12, alpha=0.55, zorder=4,
                           marker=method_marker[method], label=method if idx == 0 else "")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # shared legend on first timeline
    if labels:
        ax_first = fig.axes[1]   # first timeline axes (after table)
        handles = [
            plt.Line2D([0], [0], marker=method_marker[m], color=METHOD_COLORS[m],
                       linestyle="none", markersize=5, label=m)
            for m in method_marker
        ]
        import matplotlib.patches as mpatches
        handles.append(
            mpatches.Patch(color=PALETTE["ensemble"], alpha=0.4,
                           label="Ensemble anomaly")
        )
        ax_first.legend(handles=handles, loc="upper right", fontsize=6,
                        ncol=3, facecolor=PALETTE["panel"],
                        edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"[✓] Comparison figure saved → {output_path}")
    plt.show()

    # Save CSV
    csv_path = os.path.join(output_dir, "comparison_summary.csv")
    build_comparison_df(run_data_dict, all_results).to_csv(csv_path, index=False)
    print(f"[✓] Comparison CSV saved   → {csv_path}")

    return build_comparison_df(run_data_dict, all_results)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_runs(years: list, layers: list, results_prefix: str) -> dict:
    """
    Load Wasserstein distance series for every (year, layer) combination.

    Parameters
    ----------
    years           : list of int, e.g. [2020, 2021]
    layers          : list of str, e.g. ['contract_txs_ETH_only_750']
    results_prefix  : str, e.g. 'run_results_V8'
                      Files are expected at  {results_prefix}_{year}.json

    Returns
    -------
    dict keyed by (year, layer) -> pd.Series with DatetimeIndex
    Prints a ✓/⚠ status line for every attempted combination.
    """
    run_data = {}

    for year in years:
        fpath = f'{results_prefix}_{year}.json'
        try:
            run_results = pd.read_json(fpath)
        except FileNotFoundError:
            print(f'⚠ File not found: {fpath}')
            continue

        for layer in layers:
            if layer not in run_results.columns:
                print(f'⚠ Layer not found: {year} · {layer}')
                continue

            series = pd.Series(run_results[layer]['distance_series'])
            series.index = pd.to_datetime(series.index).normalize()
            run_data[(year, layer)] = series
            print(f'✓ Loaded: {year} · {layer} ({len(series)} days)')

    print(f'\nTotal runs loaded: {len(run_data)}')
    return run_data


print('yearly_analysis_functions.py loaded ✓')
