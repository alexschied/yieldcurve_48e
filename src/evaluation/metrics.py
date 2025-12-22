import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }
