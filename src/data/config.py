from dataclasses import dataclass
from typing import Optional, Tuple

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
