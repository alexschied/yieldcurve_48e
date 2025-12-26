import os
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

    outpath = os.path.join(outdir, f"fig_top_{metric}_{scenario_name}.png")
    plt.savefig(outpath, dpi=180)
    plt.close()
