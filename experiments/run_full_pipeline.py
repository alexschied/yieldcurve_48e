import os

import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config, define_models
from src.data import FredDataLoader

from src.features import compute_pca, make_feature_sets, rolling_pca_loadings



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

from src.visualizations import (
    plot_top_scenario_results,
    plot_pc_timeseries,
    plot_loadings_heatmap,
    plot_scree,
    plot_yield_curve,
    plot_rolling_drift
)

from src.utils import df_to_latex_table




# =========================================================
# MAIN
# =========================================================

def main():
    CFG = Config()

    os.makedirs(CFG.outdir, exist_ok=True)
    os.makedirs(CFG.tabledir, exist_ok=True)

    if "FREDAPI" not in os.environ:
        raise RuntimeError("Missing Fred API Key environment variable")

    CFG = Config()
    os.makedirs(CFG.outdir, exist_ok=True)
    os.makedirs(CFG.tabledir, exist_ok=True)

    print("============================================================")
    print("EC48E Recession Prediction — FULL PIPELINE")
    print("============================================================")
    print(f"Threshold objective = {CFG.threshold_objective} | grid_n={CFG.threshold_grid_n}\n")

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------
    loader = FredDataLoader(CFG.FredConfig())
    df_raw = loader.build_dataset(CFG.start_date, CFG.end_date)
    df = loader.add_forward_target(df_raw, CFG.horizon_months)

    print(f"Data window: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Observations (months): {len(df)}")
    print(f"Recession frequency (current month): {df['recession'].mean():.2%}")
    print(f"Target frequency (t+{CFG.horizon_months}): {df['target'].mean():.2%}\n")

    
    # ---------------------------------------------------------
    # Yield Curve Plot 2006 + 2010
    # ---------------------------------------------------------
    plot_yield_curve(df, CFG.outdir)


    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
    target_names = CFG.FredConfig().yield_series.values()
    yield_cols = [c for c in target_names if c in df.columns]

    if len(yield_cols) < 5:
        raise RuntimeError("Not enough yield columns fetched for PCA.")

    pc_df, loadings, evr = compute_pca(df, yield_cols, n_components=3)

    # --- PCA tables (LaTeX)
    df_to_latex_table(
        loadings.reset_index(),
        outpath=os.path.join(CFG.tabledir, "pca_loadings.tex"),
        caption="PCA loadings for yield curve components.",
        label="tab:pca_loadings"
    )

    df_to_latex_table(
        evr.reset_index(),
        outpath=os.path.join(CFG.tabledir, "pca_explained_variance_ratio.tex"),
        caption="Explained variance ratio of principal components.",
        label="tab:pca_evr"
    )

    # --- PCA figures
    plot_pc_timeseries(pc_df, df["recession"], CFG.outdir)
    plot_loadings_heatmap(loadings, CFG.outdir)
    plot_scree(evr, CFG.outdir)

    # --- Rolling PCA
    roll = rolling_pca_loadings(
        df,
        yield_cols,
        window_months=CFG.rolling_pca_window,
        step=CFG.rolling_pca_step
    )

    for k, v in roll.items():
        df_to_latex_table(
            v.reset_index(),
            outpath=os.path.join(CFG.tabledir, f"rolling_loadings_{k}.tex"),
            caption=f"Rolling PCA loadings ({k}).",
            label=f"tab:rolling_pca_{k}"
        )

    plot_rolling_drift(roll, CFG.outdir)
    

    # ---------------------------------------------------------
    # FEATURES & MODELS
    # ---------------------------------------------------------

    feature_sets = make_feature_sets(df, pc_df)




    # =========================================================
    # HOLDOUT SENSITIVITY
    # =========================================================

    holdout_rows = []

    for test_frac in CFG.test_size_fracs:
        for fset, dff in feature_sets.items():

            train_df, test_df = holdout_split(dff, test_size_frac=test_frac)
            if len(train_df) < 120 or len(test_df) < 12:
                continue

            Xtr = train_df.drop(columns=["target"])
            ytr = train_df["target"].astype(int)
            Xte = test_df.drop(columns=["target"])
            yte = test_df["target"].astype(int).values

            if ytr.nunique() < 2:
                continue

            model_suite = define_models(CFG)

            for mname in model_suite.get_names():

                mdl = model_suite.get_model(mname, ytr)
                

                oof_prob = oof_probs_train_only(Xtr, ytr, mname, model_suite.get_model(mname), CFG)
                thr_opt, thr_val = optimize_threshold(
                    ytr.values, oof_prob, CFG,
                    objective=CFG.threshold_objective
                )

                try:
                    mdl.fit(Xtr, ytr.values)
                    prob_te = predict_proba(mdl, Xte)
                    met = compute_metrics(yte, prob_te, thr=thr_opt)
                except Exception as e:
                    print(f"Error fitting {mname}: {e}")
                    continue

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

    df_to_latex_table(
        holdout_df,
        outpath=os.path.join(CFG.tabledir, "model_results_holdout.tex"),
        caption="Holdout sensitivity results across feature sets and models.",
        label="tab:holdout_results"
    )

    # =========================================================
    # SCENARIO TESTS
    # =========================================================

    scenario_rows = []

    for scen_name, (train_end, test_start, test_end) in CFG.scenarios.items():
        for fset, dff in feature_sets.items():

            train_df, test_df = scenario_split(
                dff, train_end, test_start, test_end
            )
            if len(train_df) < 120 or len(test_df) < 12:
                continue

            Xtr = train_df.drop(columns=["target"])
            ytr = train_df["target"].astype(int)
            Xte = test_df.drop(columns=["target"])
            yte = test_df["target"].astype(int).values

            if ytr.nunique() < 2:
                continue

            model_suite = define_models(CFG)

            for mname in model_suite.get_names():

                mdl = model_suite.get_model(mname, ytr)
                

                oof_prob = oof_probs_train_only(Xtr, ytr, mname, model_suite.get_model(mname), CFG)
                thr_opt, thr_val = optimize_threshold(
                    ytr.values, oof_prob, CFG,
                    objective=CFG.threshold_objective
                )

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

    df_to_latex_table(
        scen_df,
        outpath=os.path.join(CFG.tabledir, "model_results_scenarios.tex"),
        caption="Scenario-based evaluation results (GFC vs COVID).",
        label="tab:scenario_results"
    )

    # ---------------------------------------------------------
    # SCENARIO FIGURES
    # ---------------------------------------------------------

    if not scen_df.empty:
        for metric in ["PR_AUC", "F2"]:
            for scen_name in scen_df["Scenario"].unique():
                plot_top_scenario_results(
                    scen_df,
                    metric,
                    scen_name,
                    CFG.outdir
                )

    print(f"\nAll outputs written to:\n  figures → {CFG.outdir}\n  tables  → {CFG.tabledir}")


if __name__ == "__main__":
    main()