import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.family": "serif",
    "text.usetex": False  # set True if LaTeX installed system-wide
})


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
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_loadings_heatmap(loadings: pd.DataFrame, outpath: str):
    plt.figure(figsize=(8, 4))
    plt.imshow(loadings.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(loadings.shape[1]), loadings.columns)
    plt.yticks(range(loadings.shape[0]), loadings.index)
    plt.title("PCA Loadings Heatmap")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_scree(evr: pd.Series, outpath: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(evr) + 1), evr.values, marker="o")
    plt.xticks(range(1, len(evr) + 1), evr.index)
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
