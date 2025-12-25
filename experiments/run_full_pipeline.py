import os

import matplotlib.pyplot as plt
import pandas as pd

from src.data.config import Config
from src.data.dataset import build_dataset, add_forward_target

from src.features.pca import compute_pca
from src.features.feature_sets import make_feature_sets
from src.features.rolling_pca import (
    rolling_pca_loadings,
    plot_rolling_drift,
)

from src.models.model_defs import define_models, XGBClassifier, XGB_AVAILABLE

from src.evaluation.splits import (
    holdout_split,
    scenario_split,
)
from src.evaluation.oof import oof_probs_train_only

from src.evaluation.metrics import (
    predict_proba,
    compute_metrics,
    optimize_threshold,
)

from src.visualization.pca_plots import (
    plot_pc_timeseries,
    plot_loadings_heatmap,
    plot_scree,
)

from src.utils.misc import df_to_latex_table


if "FREDAPI" not in os.environ:
    raise RuntimeError("Missing Fred API Key environment variable")

CFG = Config()
os.makedirs(CFG.outdir, exist_ok=True)



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

    models = define_models(CFG)
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
                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj, CFG)
                thr_opt, thr_val = optimize_threshold(ytr.values, oof_prob, CFG, objective=CFG.threshold_objective)

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
                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj, CFG)
                thr_opt, thr_val = thr_opt, thr_val = optimize_threshold(ytr.values, oof_prob, CFG, objective=CFG.threshold_objective)


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
