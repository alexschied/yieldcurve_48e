import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from .metrics import predict_proba  # reuse predict_proba from metrics


def oof_probs_train_only(X_train, y_train, model_name, model_obj, CFG):
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
