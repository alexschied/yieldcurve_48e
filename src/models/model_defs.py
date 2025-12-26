from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def define_models(CFG) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["Logistic"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=CFG.random_state
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=CFG.random_state,
        n_jobs=-1
    )

    models["GradBoost"] = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=CFG.random_state
    )

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=CFG.random_state
        )

    return models

# in src/models/model_defs.py
from xgboost import XGBClassifier

__all__ = ["define_models", "XGBClassifier", "XGB_AVAILABLE"]
