import numpy as np
import pandas as pd
from typing import Tuple


def holdout_split(df: pd.DataFrame, test_size_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(np.floor((1.0 - test_size_frac) * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def scenario_split(df: pd.DataFrame,
                   train_end: str,
                   test_start: str,
                   test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train = df.loc[:pd.to_datetime(train_end)].copy()
    test = df.loc[pd.to_datetime(test_start):pd.to_datetime(test_end)].copy()
    return train, test
