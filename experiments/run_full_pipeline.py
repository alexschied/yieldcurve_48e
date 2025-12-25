import os

import matplotlib.pyplot as plt
import pandas as pd

from src.data.config import Config 
from src.data.dataset import build_dataset
from src.features.pca import compute_pca
from src.features.feature_sets import make_feature_sets
#from src.models.model_defs import define_models
#from src.evaluation.splits import holdout_split
#from src.evaluation.thresholding import optimize_threshold
#from src.evaluation.metrics import compute_metrics
from src.utils.misc import df_to_latex_table

if "FREDAPI" not in os.environ:
    raise RuntimeError("Missing Fred API Key environment variable")

CFG = Config()
os.makedirs(CFG.outdir, exist_ok=True)

# ----------------------------
# OPTIONAL: XGBoost
# ----------------------------
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ============================================================
# MODELS
# ============================================================
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score
)

def define_models() -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["Logistic"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=CFG.random_state
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=CFG.random_state,
        n_jobs=-1
    )

    models["GradBoost"] = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=CFG.random_state
    )

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=CFG.random_state
        )
    return models

# ============================================================
# TARGET (t+h)
# ============================================================
def add_forward_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["recession"]).copy()
    out["target"] = out["recession"].shift(-horizon)
    out = out.dropna(subset=["target"]).copy()
    out["target"] = out["target"].astype(int)
    out["recession"] = out["recession"].astype(int)
    return out


# ============================================================
# PCA PLOTS
# ============================================================


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
        shade_recessions(ax, rec_series, alpha=0.12)

        # highlight GFC and COVID windows
        ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"), alpha=0.15)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), alpha=0.15)

        ax.set_title(f"{col} (NBER shading + GFC/COVID highlights)")
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
    plt.title("PCA Loadings Heatmap (Maturity × PC)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_scree(evr: pd.Series, outpath: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(evr)+1), evr.values, marker="o")
    plt.xticks(range(1, len(evr)+1), evr.index)
    plt.title("Explained Variance Ratio (Scree)")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

# ============================================================
# ROLLING PCA DRIFT
# ============================================================
def rolling_pca_loadings(df: pd.DataFrame, yield_cols: List[str], window_months: int = 120, step: int = 1):
    Xfull = df[yield_cols].dropna()
    pc_loadings = {1: [], 2: [], 3: []}
    times = []

    for end_i in range(window_months, len(Xfull)+1, step):
        w = Xfull.iloc[end_i-window_months:end_i]
        t = w.index[-1]

        Xs = StandardScaler().fit_transform(w.values)
        pca = PCA(n_components=3, random_state=CFG.random_state).fit(Xs)
        L = pca.components_.T

        # sign conventions (stable drift plots)
        if L[:,0].mean() < 0: L[:,0] *= -1
        if L[-1,1] < 0:       L[:,1] *= -1
        if L[-1,2] < 0:       L[:,2] *= -1

        for k in [1,2,3]:
            pc_loadings[k].append(pd.Series(L[:,k-1], index=yield_cols))
        times.append(t)

    out = {}
    for k in [1,2,3]:
        out[f"PC{k}"] = pd.DataFrame(pc_loadings[k], index=pd.DatetimeIndex(times))
    return out

def plot_rolling_drift(roll_loadings: Dict[str, pd.DataFrame], outprefix: str):
    for pc_name, Ldf in roll_loadings.items():
        plt.figure(figsize=(12, 5))
        for col in Ldf.columns:
            plt.plot(Ldf.index, Ldf[col], label=col)

        # QE/ZLB highlight
        plt.axvspan(pd.Timestamp("2008-12-01"), pd.Timestamp("2015-12-01"), alpha=0.12)
        plt.title(f"Rolling PCA Loading Drift — {pc_name} (window={CFG.rolling_pca_window}m)")
        plt.xlabel("Window end date")
        plt.ylabel("Loading")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{outprefix}_{pc_name}.png", dpi=180)
        plt.close()


# ============================================================
# METRICS + THRESHOLDS (robust)
# ============================================================
import numpy as np
def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)
    return np.asarray(model.predict(X), dtype=float)

def fbeta_from_counts(tp, fp, fn, beta=2.0) -> float:
    beta2 = beta**2
    denom = (1+beta2)*tp + beta2*fn + fp
    return 0.0 if denom == 0 else (1+beta2)*tp/denom

def compute_metrics(y_true, y_prob, thr=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    finite = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite]

    if len(y_true) == 0:
        return {k: np.nan for k in ["Accuracy","BalancedAcc","Precision","Recall","F1","F2","ROC_AUC","PR_AUC"]}

    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_from_counts(tp, fp, fn, beta=2.0),
    }

    if len(np.unique(y_true)) == 2:
        out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        out["PR_AUC"] = average_precision_score(y_true, y_prob)
    else:
        out["ROC_AUC"] = np.nan
        out["PR_AUC"] = np.nan

    return out

def optimize_threshold(y_true, y_prob, objective="balanced_acc") -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    finite = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite]

    grid = np.linspace(CFG.threshold_min, CFG.threshold_max, CFG.threshold_grid_n)
    best_thr, best_val = 0.5, -np.inf

    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)

        if objective == "balanced_acc":
            val = balanced_accuracy_score(y_true, y_pred)
        elif objective == "f2":
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            val = fbeta_from_counts(tp, fp, fn, beta=2.0)
        else:
            raise ValueError("objective must be 'balanced_acc' or 'f2'")

        if val > best_val:
            best_val, best_thr = float(val), float(thr)

    return best_thr, best_val


def oof_probs_train_only(X_train: pd.DataFrame, y_train: pd.Series, model_name: str, model_obj) -> np.ndarray:
    """OOF probabilities inside TRAIN for threshold tuning (no leakage)."""
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train.loc[mask].copy()
    y_train = y_train.loc[mask].copy()

    tscv = TimeSeriesSplit(n_splits=CFG.tscv_splits)
    oof = np.full(len(X_train), np.nan, dtype=float)

    for tr, te in tscv.split(X_train):
        Xtr, Xte = X_train.iloc[tr], X_train.iloc[te]
        ytr = y_train.iloc[tr].values

        if len(np.unique(ytr)) < 2:
            continue

        mdl = model_obj
        if model_name == "XGBoost":
            pos = ytr.sum()
            neg = len(ytr) - pos
            spw = (neg / max(pos, 1))
            mdl = XGBClassifier(**{**model_obj.get_params(), "scale_pos_weight": spw})

        try:
            mdl.fit(Xtr, ytr)
            p = predict_proba(mdl, Xte)
            if np.any(~np.isfinite(p)):
                continue
            oof[te] = p
        except Exception:
            continue

    base = float(y_train.mean()) if len(y_train) else 0.0
    oof = np.where(np.isnan(oof), base, oof)
    return oof


# ============================================================
# VALIDATION SPLITS
# ============================================================
def holdout_split(df: pd.DataFrame, test_size_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(np.floor((1.0 - test_size_frac) * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def scenario_split(df: pd.DataFrame, train_end: str, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:pd.to_datetime(train_end)].copy()
    test  = df.loc[pd.to_datetime(test_start):pd.to_datetime(test_end)].copy()
    return train, test


def main():
    print("============================================================")
    print("EC48E Recession Prediction — FULL PIPELINE")
    print("============================================================")
    print(f"Threshold objective = {CFG.threshold_objective} | grid_n={CFG.threshold_grid_n}\n")

    df_raw = build_dataset(CFG.start_date, CFG.end_date)
    df = add_forward_target(df_raw, CFG.horizon_months)

    print(f"Data window: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Observations (months): {len(df)}")
    print(f"Recession frequency (current month): {df['recession'].mean():.2%}")
    print(f"Target frequency (t+{CFG.horizon_months}): {df['target'].mean():.2%}\n")

    # --- PCA DESCRIPTION BLOCK (Step A)
    yield_cols = [c for c in ["y_3m","y_6m","y_1y","y_2y","y_3y","y_5y","y_7y","y_10y","y_30y"] if c in df.columns]
    if len(yield_cols) < 5:
        raise RuntimeError("Not enough yield columns fetched for PCA.")

    pc_df, loadings, evr = compute_pca(df, yield_cols, n_components=3)

    # Save PCA tables
    loadings.to_csv(os.path.join(CFG.outdir, "pca_loadings_table.csv"))
    evr.to_csv(os.path.join(CFG.outdir, "pca_explained_variance_ratio.csv"))

    # PCA plots
    plot_pc_timeseries(pc_df, df["recession"], os.path.join(CFG.outdir, "fig_PC_timeseries.png"))
    plot_loadings_heatmap(loadings, os.path.join(CFG.outdir, "fig_PCA_loadings_heatmap.png"))
    plot_scree(evr, os.path.join(CFG.outdir, "fig_PCA_scree.png"))

    # Rolling PCA drift (optional but powerful)
    roll = rolling_pca_loadings(df, yield_cols, window_months=CFG.rolling_pca_window, step=CFG.rolling_pca_step)
    for k, v in roll.items():
        v.to_csv(os.path.join(CFG.outdir, f"rolling_loadings_{k}.csv"))
    plot_rolling_drift(roll, os.path.join(CFG.outdir, "fig_rolling_loading_drift"))

    # --- Feature sets (Step B/C)
    feature_sets = make_feature_sets(df, pc_df)
    print(f"Feature sets created: {list(feature_sets.keys())}\n")

    models = define_models()
    print(f"Models: {list(models.keys())}\n")

    # =========================================================
    # HOLDOUT SENSITIVITY RUNS
    # =========================================================
    holdout_rows = []

    for test_frac in CFG.test_size_fracs:
        print("============================================================")
        print(f"HOLDOUT SENSITIVITY RUN: test_size_frac={test_frac:.2f}")
        print("============================================================")

        for fset, dff in feature_sets.items():
            train_df, test_df = holdout_split(dff, test_size_frac=test_frac)

            Xtr = train_df.drop(columns=["target"]).copy()
            ytr = train_df["target"].astype(int).copy()
            Xte = test_df.drop(columns=["target"]).copy()
            yte = test_df["target"].astype(int).values

            if len(train_df) < 120 or len(test_df) < 12:
                continue
            if ytr.nunique() < 2:
                continue

            print("\n------------------------------------------------------------")
            print(f"Feature set: {fset} | train={len(train_df)} test={len(test_df)} | pos(train)={ytr.mean():.2%}")

            for mname, mobj in models.items():
                # threshold on TRAIN only using OOF
                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj)
                thr_opt, thr_val = optimize_threshold(ytr.values, oof_prob, objective=CFG.threshold_objective)

                mdl = mobj
                if mname == "XGBoost":
                    pos = ytr.values.sum()
                    neg = len(ytr) - pos
                    spw = (neg / max(pos, 1))
                    mdl = XGBClassifier(**{**mobj.get_params(), "scale_pos_weight": spw})

                try:
                    mdl.fit(Xtr, ytr.values)
                    prob_te = predict_proba(mdl, Xte)
                    met = compute_metrics(yte, prob_te, thr=thr_opt)
                except Exception:
                    continue

                print(f"  {mname:12s} | thr={thr_opt:.3f} | "
                      f"ROC_AUC={met['ROC_AUC']:.3f} | PR_AUC={met['PR_AUC']:.3f} | "
                      f"BalAcc={met['BalancedAcc']:.3f} | F2={met['F2']:.3f}")

                holdout_rows.append({
                    "TestFrac": test_frac,
                    "FeatureSet": fset,
                    "Model": mname,
                    "ThrObjective": CFG.threshold_objective,
                    "ThrOpt": thr_opt,
                    "ThrObjectiveValue_onTrainOOF": thr_val,
                    **met
                })

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_path = os.path.join(CFG.outdir, "model_results_holdout.csv")
    holdout_df.to_csv(holdout_path, index=False)
    print(f"\nSaved holdout results: {holdout_path}")

    # =========================================================
    # SCENARIO TESTS: GFC vs COVID (your “convince prof” section)
    # =========================================================
    scenarios = {
        "GFC_test_2007_2009": ("2006-12-01", "2007-01-01", "2009-12-01"),
        "COVID_test_2019_2021": ("2018-12-01", "2019-01-01", "2021-12-01"),
    }

    scenario_rows = []

    print("\n============================================================")
    print("SCENARIO TESTS (GFC vs COVID) — time-respecting splits")
    print("============================================================")

    for scen_name, (train_end, test_start, test_end) in scenarios.items():
        print(f"\n--- Scenario: {scen_name} | train<= {train_end} | test: {test_start}..{test_end}")

        for fset, dff in feature_sets.items():
            train_df, test_df = scenario_split(dff, train_end, test_start, test_end)
            if len(train_df) < 120 or len(test_df) < 12:
                continue

            Xtr = train_df.drop(columns=["target"]).copy()
            ytr = train_df["target"].astype(int).copy()
            Xte = test_df.drop(columns=["target"]).copy()
            yte = test_df["target"].astype(int).values

            if ytr.nunique() < 2:
                continue

            for mname, mobj in models.items():
                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj)
                thr_opt, thr_val = optimize_threshold(ytr.values, oof_prob, objective=CFG.threshold_objective)

                mdl = mobj
                if mname == "XGBoost":
                    pos = ytr.values.sum()
                    neg = len(ytr) - pos
                    spw = (neg / max(pos, 1))
                    mdl = XGBClassifier(**{**mobj.get_params(), "scale_pos_weight": spw})

                try:
                    mdl.fit(Xtr, ytr.values)
                    prob_te = predict_proba(mdl, Xte)
                    met = compute_metrics(yte, prob_te, thr=thr_opt)
                except Exception:
                    continue

                scenario_rows.append({
                    "Scenario": scen_name,
                    "TrainEnd": train_end,
                    "TestStart": test_start,
                    "TestEnd": test_end,
                    "FeatureSet": fset,
                    "Model": mname,
                    "ThrObjective": CFG.threshold_objective,
                    "ThrOpt": thr_opt,
                    "ThrObjectiveValue_onTrainOOF": thr_val,
                    **met
                })

    scen_df = pd.DataFrame(scenario_rows)
    scen_path = os.path.join(CFG.outdir, "model_results_scenarios.csv")
    scen_df.to_csv(scen_path, index=False)
    print(f"\nSaved scenario results: {scen_path}")

    # Quick “prof-friendly” plots: PR_AUC and F2 by scenario
    if not scen_df.empty:
        for metric in ["PR_AUC", "F2"]:
            for scen_name in scen_df["Scenario"].unique():
                sub = scen_df[scen_df["Scenario"] == scen_name].copy()
                if sub.empty:
                    continue

                # best per (featureset, model) already; just plot all
                sub = sub.sort_values(metric, ascending=False).head(15)

                plt.figure(figsize=(10, 5))
                labels = sub["Model"] + " | " + sub["FeatureSet"]
                plt.bar(range(len(sub)), sub[metric].values)
                plt.xticks(range(len(sub)), labels, rotation=45, ha="right", fontsize=8)
                plt.title(f"Top results by {metric} — {scen_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(CFG.outdir, f"fig_top_{metric}_{scen_name}.png"), dpi=180)
                plt.close()

    print(f"\nAll outputs in: {CFG.outdir}")


if __name__ == "__main__":
    main()
