import os
from src.data.config import Config 
from src.data.dataset import build_dataset
from src.features.pca import compute_pca
from src.features.feature_sets import make_feature_sets
from src.models.model_defs import define_models
from src.evaluation.splits import holdout_split
from src.evaluation.thresholding import optimize_threshold
from src.evaluation.metrics import compute_metrics

def main():
    cfg = Config()
    os.makedirs(cfg.outdir, exist_ok=True)

    df = build_dataset(cfg.start_date, cfg.end_date)
    df["target"] = df["recession"].shift(-cfg.horizon_months)
    df = df.dropna()

    yield_cols = [c for c in df.columns if c.startswith("y_")]
    pc_df, loadings, evr = compute_pca(df, yield_cols)

    fsets = make_feature_sets(df, pc_df)
    models = define_models(cfg.random_state)

    for name, dff in fsets.items():
        train, test = holdout_split(dff, 0.3)
        Xtr, ytr = train.drop("target", axis=1), train["target"]
        Xte, yte = test.drop("target", axis=1), test["target"]

        for mname, model in models.items():
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:,1]
            thr = optimize_threshold(yte, prob)
            print(name, mname, compute_metrics(yte, prob, thr))

if __name__ == "__main__":
    main()

from src.utils.misc import df_to_latex_table

# After you load results CSV
df = pd.read_csv("outputs/ec48e_outputs_full/model_results_holdout.csv")

# Example: best models by PR_AUC
tbl = (
    df.sort_values("PR_AUC", ascending=False)
      .groupby(["FeatureSet", "Model"])
      .head(1)
      [["FeatureSet", "Model", "PR_AUC", "F2", "BalancedAcc"]]
)

df_to_latex_table(
    tbl,
    outpath="reports/tables/holdout_results.tex",
    caption="Holdout performance across feature sets and models.",
    label="tab:holdout-results"
)
