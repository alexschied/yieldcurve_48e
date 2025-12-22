import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def oof_probs_train_only(model, X, y, splits):
    tscv = TimeSeriesSplit(splits)
    oof = np.full(len(X), np.nan)

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:,1]

    return oof
