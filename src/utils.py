import pandas as pd
from pathlib import Path

def extract_importance(model, feature_names, model_name):
    """
    Extrahiert Feature-Wichtigkeit basierend auf dem Modelltyp.
    """
    import numpy as np
    import pandas as pd

    # Pipeline-Extraktion (für Logistic, Ridge, ElasticNet)
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['clf']
        # Spezialfall für CalibratedClassifierCV (Ridge)
        if hasattr(clf, 'calibrated_classifiers_'):
            # Durchschnitt der Koeffizienten der internen Klassifikatoren
            coefs = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
        else:
            coefs = clf.coef_
        importance = np.abs(coefs).flatten()
    
    # Baum-Modelle (RF, GradBoost, XGBoost)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    else:
        return pd.Series()

    return pd.Series(importance, index=feature_names)

def df_to_latex_table(
    df: pd.DataFrame,
    outpath: str,
    caption: str,
    label: str,
    float_format="%.3f",
    longtable: bool = False
):
    """
    Export a publication-ready LaTeX table.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    latex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        longtable=longtable,
        escape=True,
        column_format="l" + "c" * (len(df.columns) - 1)
    )

    with open(outpath, "w") as f:
        f.write(latex)
