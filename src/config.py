from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union, Any

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


@dataclass
class Config:
    start_date: str = "1982-01-01"
    end_date: Optional[str] = None

    horizon_months: int = 12
    test_size_fracs: Tuple[float, ...] = (0.20, 0.30, 0.40)
    tscv_splits: int = 6

    rolling_pca_window: int = 120
    rolling_pca_step: int = 1

    threshold_objective: str = "balanced_acc"
    threshold_grid_n: int = 201
    threshold_min: float = 0.001
    threshold_max: float = 0.999

    outdir: str = "reports/figures"
    tabledir: str = "reports/tables"
    random_state: int = 187

    scenarios = {
        "GFC_test_2007_2009": ("2006-12-01", "2007-01-01", "2009-12-01"),
        "COVID_test_2019_2021": ("2018-12-01", "2019-01-01", "2021-12-01"),
    }

    @dataclass
    class FredConfig:
        """
        Configuration for FRED data loading, including series IDs and regime definitions.
        """

        api_key_env_var: str = "FREDAPI"
        
        # Map FRED Series ID -> DataFrame Column Name
        yield_series: Dict[str, str] = field(default_factory=lambda: {
            "DGS3MO": "y_3m", "DGS6MO": "y_6m", "DGS1": "y_1y",
            "DGS2": "y_2y", "DGS3": "y_3y", "DGS5": "y_5y",
            "DGS7": "y_7y", "DGS10": "y_10y", "DGS30": "y_30y"
        })
        
        macro_series: Dict[str, str] = field(default_factory=lambda: {
            "UNRATE": "unrate", 
            "INDPRO": "indpro",
            "UMCSENT": "umcsent", 
            "CPIAUCSL": "cpi"
        })
        
        recession_series_id: str = "USREC"
        recession_col_name: str = "recession"
        
        # Map Column Name -> (Start Date, End Date)
        regime_definitions: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
            "D_GFC": ("2007-12-01", "2009-06-01"),
            "D_COVID": ("2020-03-01", "2020-05-01"),
            "D_ZLB_QE": ("2008-12-01", "2015-12-01")
        })



@dataclass(frozen=True)
class ModelSuite:
    logistic: Pipeline
    random_forest: RandomForestClassifier
    grad_boost: GradientBoostingClassifier
    xgboost: Optional[Any] = None

    def get_model(self, name: str, y_train: pd.Series = None) -> Any:
        """
        Returns a specific model by name. 
        If it's XGBoost, it automatically handles scale_pos_weight based on y_train.
        """
        models = {
            "Logistic": self.logistic,
            "RandomForest": self.random_forest,
            "GradBoost": self.grad_boost,
            "XGBoost": self.xgboost
        }
        
        mdl = models.get(name)
        if mdl is None:
            return None

        # Special handling for XGBoost scaling
        if name == "XGBoost" and y_train is not None:
            pos = y_train.sum()
            neg = len(y_train) - pos
            # We return a clone/new instance with the specific weight for this fold
            params = mdl.get_params()
            params["scale_pos_weight"] = neg / max(pos, 1)
            return XGBClassifier(**params)
            
        return mdl

    def get_names(self) -> list:
        names = ["Logistic", "RandomForest", "GradBoost"]
        if self.xgboost:
            names.append("XGBoost")
        return names

def define_models(CFG) -> ModelSuite:
    logistic = Pipeline([
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

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=CFG.random_state,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=CFG.random_state
    )

    xgb = None
    if XGB_AVAILABLE:
        xgb = XGBClassifier(
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

    return ModelSuite(
        logistic=logistic,
        random_forest=rf,
        grad_boost=gb,
        xgboost=xgb
    )