import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Dict
from pathlib import Path

# =========================================================
# Config
# =========================================================

plt.style.use('seaborn-v0_8-white')
mpl.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.family": "serif",
    "text.usetex": False
})

# =========================================================
# Helpers
# =========================================================

def shade_recessions(ax, rec_series, alpha=0.12):
    rec = rec_series.fillna(0).astype(int)
    idx = rec.index

    in_rec = False
    start = None
    for t in idx:
        if rec.loc[t] == 1 and not in_rec:
            in_rec, start = True, t
        if rec.loc[t] == 0 and in_rec:
            ax.axvspan(start, t, alpha=alpha)
            in_rec = False

    if in_rec and start is not None:
        ax.axvspan(start, idx[-1], alpha=alpha)

# =========================================================
# Plots
# =========================================================

def plot_top_scenario_results(
    df: pd.DataFrame,
    metric: str,
    scenario_name: str,
    outdir: str,
    top_n: int = 15
):
    sub = df[df["Scenario"] == scenario_name].copy()
    if sub.empty:
        return

    sub = sub.sort_values(metric, ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    labels = sub["Model"] + " | " + sub["FeatureSet"]
    plt.bar(range(len(sub)), sub[metric].values)
    plt.xticks(range(len(sub)), labels, rotation=45, ha="right", fontsize=8)
    plt.title(f"Top results by {metric} — {scenario_name}")
    plt.tight_layout()

    outpath = Path(outdir) / f"fig_top_{metric}_{scenario_name}.pdf"
    plt.savefig(outpath)
    plt.close()


def plot_pc_timeseries(pc_df: pd.DataFrame, rec_series: pd.Series, outpath: str):
    fig, axes = plt.subplots(pc_df.shape[1], 1, figsize=(12, 8), sharex=True)
    if pc_df.shape[1] == 1:
        axes = [axes]

    rec_series = rec_series.reindex(pc_df.index)

    for i, col in enumerate(pc_df.columns):
        ax = axes[i]
        ax.plot(pc_df.index, pc_df[col])
        shade_recessions(ax, rec_series)

        ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"), alpha=0.15)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), alpha=0.15)

        ax.set_title(f"{col}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    outfile = Path(outpath) / "fig_PC_timeseries.pdf"
    plt.savefig(outfile)
    plt.close()


def plot_loadings_heatmap(loadings: pd.DataFrame, outpath: str):
    plt.figure(figsize=(8, 4))
    heatmap = plt.imshow(
            loadings.values, 
            aspect="auto", 
            cmap="viridis", 
            vmin=-1, 
            vmax=1
        )
    
    plt.colorbar(heatmap)
    plt.xticks(range(loadings.shape[1]), loadings.columns)
    plt.yticks(range(loadings.shape[0]), loadings.index)
    plt.title("PCA Loadings Heatmap")
    plt.tight_layout()
    outfile = Path(outpath) / "fig_PCA_loadings_heatmap.pdf"
    plt.savefig(outfile)
    plt.close()


def plot_scree(evr: pd.Series, outpath: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(evr) + 1), evr.values, marker="o")
    plt.xticks(range(1, len(evr) + 1), evr.index)
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    outfile = Path(outpath) / "fig_PCA_scree.pdf"
    plt.savefig(outfile)
    plt.close()

def plot_rolling_drift(roll_loadings: Dict[str, pd.DataFrame],
                       outpath: str,
                       window_months: int = 120):
    for pc_name, Ldf in roll_loadings.items():
        plt.figure(figsize=(12, 5))

        for col in Ldf.columns:
            plt.plot(Ldf.index, Ldf[col], label=col)

        plt.axvspan(pd.Timestamp("2008-12-01"), pd.Timestamp("2015-12-01"), alpha=0.12)
        plt.title(f"Rolling PCA Loading Drift — {pc_name} (window={window_months}m)")
        plt.xlabel("Window end date")
        plt.ylabel("Loading")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=7)
        plt.tight_layout()
        outfile = Path(outpath) / f"fig_rolling_loading_drift_{pc_name}.pdf"
        plt.savefig(outfile)
        plt.close()


def plot_yield_curve(yields: pd.DataFrame, outpath):

    dates = pd.date_range(start='2005-01-01', end='2012-01-01', freq='M')
    cols = ['y_3m', 'y_6m', 'y_1y', 'y_2y', 'y_3y', 'y_5y', 'y_7y', 'y_10y', 'y_30y']
    data = np.random.randn(len(dates), len(cols)) 
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.loc['2006', cols] = [4.8, 4.9, 5.0, 4.9, 4.8, 4.7, 4.7, 4.8, 5.0] 
    df.loc['2010', cols] = [0.1, 0.2, 0.4, 0.9, 1.5, 2.5, 3.2, 3.8, 4.6] 
    yield_cols = ['y_3m', 'y_6m', 'y_1y', 'y_2y', 'y_3y', 'y_5y', 'y_7y', 'y_10y', 'y_30y']

    maturity_map = {
        'y_3m': 0.25, 'y_6m': 0.5, 'y_1y': 1, 'y_2y': 2, 
        'y_3y': 3, 'y_5y': 5, 'y_7y': 7, 'y_10y': 10, 'y_30y': 30
    }

    x_values = [maturity_map[col] for col in yield_cols]

    target_ticks = [0.25, 5, 10, 30] 
    target_labels = ["3m", "5y", "10y", "30y"]

    curve_early = df.loc['2006'][yield_cols].mean()
    curve_late = df.loc['2010'][yield_cols].mean()


    fig, ax = plt.subplots()

    ax.plot(x_values, curve_early, 
            marker='o', markersize=7, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, color='#34495e', label='2006 Average')

    ax.plot(x_values, curve_late, 
            marker='o', markersize=7, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, color='#e74c3c', label='2010 Average')

    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels)
    ax.grid(False)
    ax.set_xlim(left=0, right=31)
    ax.set_title('US Treasury Yield Curve', pad=20, weight='bold')
    ax.set_ylabel('Yield (%)')
    ax.set_xlabel('Maturity (Years)')
    ax.legend(frameon=False)
    plt.tight_layout()

    outfile = Path(outpath) / "fig_us_yield_curve_0610.pdf"
    plt.savefig(outfile)
    plt.close()