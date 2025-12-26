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
from src.visualization.results_plots import plot_top_scenario_results

from src.utils.misc import df_to_latex_table


if "FREDAPI" not in os.environ:
    raise RuntimeError("Missing Fred API Key environment variable")

CFG = Config()
os.makedirs(CFG.outdir, exist_ok=True)


# =========================================================
# MAIN
# =========================================================

def main():

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

    df_raw = build_dataset(CFG.start_date, CFG.end_date)
    df = add_forward_target(df_raw, CFG.horizon_months)

    print(f"Data window: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Observations (months): {len(df)}")
    print(f"Recession frequency (current month): {df['recession'].mean():.2%}")
    print(f"Target frequency (t+{CFG.horizon_months}): {df['target'].mean():.2%}\n")

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------

    yield_cols = [
        c for c in [
            "y_3m","y_6m","y_1y","y_2y","y_3y",
            "y_5y","y_7y","y_10y","y_30y"
        ] if c in df.columns
    ]

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
    plot_pc_timeseries(pc_df, df["recession"],
        os.path.join(CFG.outdir, "fig_PC_timeseries.png"))
    plot_loadings_heatmap(loadings,
        os.path.join(CFG.outdir, "fig_PCA_loadings_heatmap.png"))
    plot_scree(evr,
        os.path.join(CFG.outdir, "fig_PCA_scree.png"))

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

    plot_rolling_drift(
        roll,
        os.path.join(CFG.outdir, "fig_rolling_loading_drift")
    )

    # ---------------------------------------------------------
    # FEATURES & MODELS
    # ---------------------------------------------------------

    feature_sets = make_feature_sets(df, pc_df)
    models = define_models(CFG)

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

            for mname, mobj in models.items():

                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj, CFG)
                thr_opt, thr_val = optimize_threshold(
                    ytr.values, oof_prob, CFG,
                    objective=CFG.threshold_objective
                )

                mdl = mobj
                if mname == "XGBoost":
                    pos = ytr.sum()
                    neg = len(ytr) - pos
                    mdl = XGBClassifier(
                        **{**mobj.get_params(),
                           "scale_pos_weight": neg / max(pos, 1)}
                    )

                try:
                    mdl.fit(Xtr, ytr.values)
                    prob_te = predict_proba(mdl, Xte)
                    met = compute_metrics(yte, prob_te, thr=thr_opt)
                except Exception:
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

    scenarios = {
        "GFC_test_2007_2009": ("2006-12-01", "2007-01-01", "2009-12-01"),
        "COVID_test_2019_2021": ("2018-12-01", "2019-01-01", "2021-12-01"),
    }

    scenario_rows = []

    for scen_name, (train_end, test_start, test_end) in scenarios.items():
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

            for mname, mobj in models.items():

                oof_prob = oof_probs_train_only(Xtr, ytr, mname, mobj, CFG)
                thr_opt, thr_val = optimize_threshold(
                    ytr.values, oof_prob, CFG,
                    objective=CFG.threshold_objective
                )

                mdl = mobj
                if mname == "XGBoost":
                    pos = ytr.sum()
                    neg = len(ytr) - pos
                    mdl = XGBClassifier(
                        **{**mobj.get_params(),
                           "scale_pos_weight": neg / max(pos, 1)}
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