from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def rolling_pca_loadings(df: pd.DataFrame, yield_cols: List[str],
                         window_months: int = 120, step: int = 1,
                         random_state: int | None = None):


    Xfull = df[yield_cols].dropna()
    pc_loadings = {1: [], 2: [], 3: []}
    times = []

    for end_i in range(window_months, len(Xfull) + 1, step):
        w = Xfull.iloc[end_i - window_months:end_i]
        t = w.index[-1]

        Xs = StandardScaler().fit_transform(w.values)
        pca = PCA(n_components=3, random_state=random_state).fit(Xs)
        L = pca.components_.T

        if L[:, 0].mean() < 0: L[:, 0] *= -1
        if L[-1, 1] < 0: L[:, 1] *= -1
        if L[-1, 2] < 0: L[:, 2] *= -1

        for k in [1, 2, 3]:
            pc_loadings[k].append(pd.Series(L[:, k - 1], index=yield_cols))
        times.append(t)

    return {
        f"PC{k}": pd.DataFrame(pc_loadings[k], index=pd.DatetimeIndex(times))
        for k in [1, 2, 3]
    }


def plot_rolling_drift(roll_loadings: Dict[str, pd.DataFrame],
                       outprefix: str,
                       window_months: int = 120):
    for pc_name, Ldf in roll_loadings.items():
        plt.figure(figsize=(12, 5))

        for col in Ldf.columns:
            plt.plot(Ldf.index, Ldf[col], label=col)

        # QE/ZLB highlight
        plt.axvspan(pd.Timestamp("2008-12-01"), pd.Timestamp("2015-12-01"), alpha=0.12)
        plt.title(f"Rolling PCA Loading Drift — {pc_name} (window={window_months}m)")
        plt.xlabel("Window end date")
        plt.ylabel("Loading")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{outprefix}_{pc_name}.png", dpi=180)
        plt.close()

