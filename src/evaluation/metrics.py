from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, average_precision_score
)


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def fbeta_from_counts(tp, fp, fn, beta=2.0):
    beta2 = beta ** 2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return 0.0 if denom == 0 else (1 + beta2) * tp / denom


def compute_metrics(y_true, y_prob, thr=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= thr).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_from_counts(tp, fp, fn, beta=2.0),
    }

    if len(set(y_true)) == 2:
        out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        out["PR_AUC"] = average_precision_score(y_true, y_prob)
    else:
        out["ROC_AUC"] = np.nan
        out["PR_AUC"] = np.nan

    return out


def optimize_threshold(y_true, y_prob, CFG, objective="balanced_acc") -> Tuple[float, float]:
    grid = np.linspace(CFG.threshold_min, CFG.threshold_max, CFG.threshold_grid_n)
    best_thr, best_val = 0.5, -np.inf

    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)

        if objective == "balanced_acc":
            val = balanced_accuracy_score(y_true, y_pred)
        else:
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            val = fbeta_from_counts(tp, fp, fn, beta=2.0)

        if val > best_val:
            best_val, best_thr = val, thr

    return float(best_thr), float(best_val)
