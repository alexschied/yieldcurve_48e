# main.py
import os
import hashlib
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config, define_models
from src.data import FredDataLoader
from src.features import (
    compute_pca, make_feature_sets, rolling_pca_loadings
)
from src.evaluation.splits import holdout_split, scenario_split
from src.evaluation.oof import oof_probs_train_only
from src.evaluation.metrics import predict_proba, compute_metrics, optimize_threshold
from src.visualizations import (
    plot_top_scenario_results, plot_pc_timeseries, plot_loadings_heatmap,
    plot_scree, plot_yield_curve, plot_rolling_drift
)
from src.utils import df_to_latex_table

# ---------------------------------------------------------
# CACHING LOGIC
# ---------------------------------------------------------
def get_cache_path(mname: str, fset: str, Xtr: pd.DataFrame, ytr: pd.Series, CFG: Config, scenario_id: str):
    data_sig = hashlib.md5(
        f"{list(Xtr.columns)}_{Xtr.shape}_{ytr.sum()}_{CFG.threshold_objective}_{CFG.random_state}".encode()
    ).hexdigest()[:12]
    
    cache_dir = "cache/models"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{scenario_id}_{fset}_{mname}_{data_sig}.joblib")


import shap

def plot_feature_interpretation(model, X, mname, scen_name, outdir):
    """
    Generates SHAP summary plots to show which features drive recession predictions.
    """
    plt.figure()
    # Use TreeExplainer for XGB/RF, LinearExplainer for Logistic
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"Feature Importance: {mname} ({scen_name})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_{mname}_{scen_name}.pdf"))
    plt.close()

import seaborn as sns

def plot_dynamic_loadings_heatmap(roll_dict, outdir):
    """
    Creates a heatmap for each PC showing how maturity loadings drift over time.
    """
    for pc_name, df_roll in roll_dict.items():
        plt.figure(figsize=(12, 6))
        # Transpose so Y-axis is Maturity and X-axis is Time
        sns.heatmap(df_roll.T, cmap="RdBu_r", center=0)
        plt.title(f"Dynamic Loading Drift: {pc_name}")
        plt.xlabel("Window End Date")
        plt.ylabel("Maturity")
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"dynamic_heatmap_{pc_name}.pdf"))
        plt.close()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    CFG = Config()
    os.makedirs(CFG.outdir, exist_ok=True)
    os.makedirs(CFG.tabledir, exist_ok=True)

    if "FREDAPI" not in os.environ:
        raise RuntimeError("Missing Fred API Key")

    # Data & PCA
    loader = FredDataLoader(CFG.FredConfig())
    df_raw = loader.build_dataset(CFG.start_date, CFG.end_date)
    df = loader.add_forward_target(df_raw, CFG.horizon_months)
    plot_yield_curve(df, CFG.outdir)

    target_names = CFG.FredConfig().yield_series.values()
    yield_cols = [c for c in target_names if c in df.columns]
    pc_df, loadings, evr = compute_pca(df, yield_cols, n_components=3)

    # Plot PCA
    plot_pc_timeseries(pc_df, df["recession"], CFG.outdir)
    plot_loadings_heatmap(loadings, CFG.outdir)
    plot_scree(evr, CFG.outdir)
    
    roll = rolling_pca_loadings(df, yield_cols, CFG.rolling_pca_window, CFG.rolling_pca_step)
    plot_rolling_drift(roll, CFG.outdir)
    plot_dynamic_loadings_heatmap(roll, CFG.outdir)
    
    feature_sets = make_feature_sets(df, pc_df)

    # Holdout Sensitivity Loop
    holdout_rows = []
    for test_frac in CFG.test_size_fracs:
        for fset, dff in feature_sets.items():
            train_df, test_df = holdout_split(dff, test_size_frac=test_frac)
            if len(train_df) < 120: continue
            
            Xtr, ytr = train_df.drop(columns=["target"]), train_df["target"].astype(int)
            Xte, yte = test_df.drop(columns=["target"]), test_df["target"].astype(int).values
            
            suite = define_models(CFG)
            scen_id = f"holdout_{int(test_frac*100)}"

            for mname in suite.get_names():
                cp = get_cache_path(mname, fset, Xtr, ytr, CFG, scen_id)
                if os.path.exists(cp):
                    c = joblib.load(cp)
                    mdl, thr_opt, thr_val = c['mdl'], c['thr_opt'], c['thr_val']
                else:
                    mdl = suite.get_model(mname, ytr)
                    oof = oof_probs_train_only(Xtr, ytr, mname, suite.get_model(mname), CFG)
                    thr_opt, thr_val = optimize_threshold(ytr.values, oof, CFG, CFG.threshold_objective)
                    mdl.fit(Xtr, ytr.values)
                    joblib.dump({'mdl': mdl, 'thr_opt': thr_opt, 'thr_val': thr_val}, cp)

                prob_te = predict_proba(mdl, Xte)
                holdout_rows.append({"TestFrac": test_frac, "FeatureSet": fset, "Model": mname, **compute_metrics(yte, prob_te, thr_opt)})

    # Scenario Tests Loop
    scenario_rows = []
    for scen_name, (tr_e, te_s, te_e) in CFG.scenarios.items():
        for fset, dff in feature_sets.items():
            train_df, test_df = scenario_split(dff, tr_e, te_s, te_e)
            if len(train_df) < 120: continue

            Xtr, ytr = train_df.drop(columns=["target"]), train_df["target"].astype(int)
            Xte, yte = test_df.drop(columns=["target"]), test_df["target"].astype(int).values

            suite = define_models(CFG)
            for mname in suite.get_names():
                cp = get_cache_path(mname, fset, Xtr, ytr, CFG, scen_name)
                if os.path.exists(cp):
                    c = joblib.load(cp)
                    mdl, thr_opt = c['mdl'], c['thr_opt']
                else:
                    mdl = suite.get_model(mname, ytr)
                    oof = oof_probs_train_only(Xtr, ytr, mname, suite.get_model(mname), CFG)
                    thr_opt, _ = optimize_threshold(ytr.values, oof, CFG, CFG.threshold_objective)
                    mdl.fit(Xtr, ytr.values)
                    joblib.dump({'mdl': mdl, 'thr_opt': thr_opt}, cp)

                #plot_feature_interpretation(mdl, Xtr, mname, scen_name, CFG.outdir)
                prob_te = predict_proba(mdl, Xte)
                scenario_rows.append({"Scenario": scen_name, "FeatureSet": fset, "Model": mname, **compute_metrics(yte, prob_te, thr_opt)})

    # Final Outputs
    scen_df = pd.DataFrame(scenario_rows)
    df_to_latex_table(pd.DataFrame(holdout_rows), os.path.join(CFG.tabledir, "holdout.tex"), "Holdout", "tab:h", longtable=True)
    df_to_latex_table(scen_df, os.path.join(CFG.tabledir, "scenarios.tex"), "Scenarios", "tab:s")

    if not scen_df.empty:
        for scen_name in scen_df["Scenario"].unique():
            plot_top_scenario_results(scen_df, "PR_AUC", scen_name, CFG.outdir)

    print(f"Pipeline finished. Data saved in {CFG.outdir} and {CFG.tabledir}")

if __name__ == "__main__":
    main()