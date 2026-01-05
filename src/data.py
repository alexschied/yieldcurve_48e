import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union

import pandas as pd
from fredapi import Fred

@dataclass
class FredConfig:
    ###
    # Configuration for FRED data loading, including series IDs and regime definitions.
    ###
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


class FredDataLoader:
    """
    A modular class to fetch, clean, and feature-engineer economic data from FRED.
    """
    def __init__(self, config: Optional[FredConfig] = None):
        self.config = config if config else FredConfig()
        self.fred = self._init_client()

    def _init_client(self) -> Fred:
        """Initializes the Fred API client."""
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(f"Environment variable '{self.config.api_key_env_var}' not set.")
        return Fred(api_key=api_key)

    def fetch_series_monthly(self, series_id: str, start: str, end: str) -> pd.Series:
        """Fetches a single series, ensures datetime index, and resamples to monthly mean."""
        try:
            s = self.fred.get_series(series_id, observation_start=start, observation_end=end)
            if s is None or s.empty:
                 print(f"Warning: No data returned for {series_id}")
                 return pd.Series(dtype=float)
            
            s = pd.Series(s)
            s.index = pd.to_datetime(s.index)
            return s.resample("ME").mean()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch series {series_id}: {e}")

    def _normalize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Snaps index dates to the beginning of the month."""
        df = df.copy()
        df.index = (df.index + pd.offsets.MonthBegin(0)).normalize()
        return df

    def _add_regime_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds dummy columns for specific date ranges defined in config."""
        df = df.copy()
        idx = df.index

        for col_name, (start, end) in self.config.regime_definitions.items():
            # Create a boolean mask and convert to int
            mask = (idx >= start) & (idx <= end)
            df[col_name] = mask.astype(int)
            
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates specific financial spreads and YoY growth rates."""
        df = df.copy()
        
        # NOTE: These calculations depend on specific column names existing.
        # We check existence to prevent KeyErrors if config changes.
        
        # Year-over-Year Inflation
        if "cpi" in df.columns:
            df["infl_yoy"] = 100 * (df["cpi"] / df["cpi"].shift(12) - 1)
            
        # Year-over-Year Industrial Production
        if "indpro" in df.columns:
            df["indpro_yoy"] = 100 * (df["indpro"] / df["indpro"].shift(12) - 1)
            
        # Yield Spreads
        if "y_10y" in df.columns and "y_3m" in df.columns:
            df["spread_10y_3m"] = df["y_10y"] - df["y_3m"]
            
        if "y_10y" in df.columns and "y_2y" in df.columns:
            df["spread_10y_2y"] = df["y_10y"] - df["y_2y"]
            
        return df

    def build_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Orchestrates the fetching and processing of all configured data.
        """
        frames = []

        # 1. Fetch Yield Series
        for sid, name in self.config.yield_series.items():
            s = self.fetch_series_monthly(sid, start_date, end_date)
            s.name = name
            frames.append(s)

        # 2. Fetch Recession Indicator (Special handling for boolean logic)
        rec = self.fetch_series_monthly(self.config.recession_series_id, start_date, end_date)
        rec = (rec > 0.5).astype(int)
        rec.name = self.config.recession_col_name
        frames.append(rec)

        # 3. Fetch Macro Series
        for sid, name in self.config.macro_series.items():
            s = self.fetch_series_monthly(sid, start_date, end_date)
            s.name = name
            frames.append(s)

        # 4. Merge and Align
        if not frames:
            return pd.DataFrame()
            
        df = pd.concat(frames, axis=1).sort_index()
        df = self._normalize_index(df)

        # 5. Feature Engineering
        df = self._add_derived_features(df)
        df = self._add_regime_dummies(df)

        return df

    @staticmethod
    def add_forward_target(df: pd.DataFrame, horizon: int, target_col: str = "recession") -> pd.DataFrame:
        """
        Adds a forward-looking target variable shifted by 'horizon' periods.
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column {target_col} not found in DataFrame.")

        out = df.copy()
        # Ensure we have valid current values
        out = out.dropna(subset=[target_col])

        # Create target
        out["target"] = out[target_col].shift(-horizon)
        
        # Drop rows where target is NaN (the last 'horizon' rows)
        out = out.dropna(subset=["target"])

        out["target"] = out["target"].astype(int)
        out[target_col] = out[target_col].astype(int)
        
        return out

# --- Usage Example ---
if __name__ == "__main__":
    # Ensure you have set export FREDAPI='your_key_here' in your terminal
    
    try:
        loader = FredDataLoader()
        
        print("Fetching data...")
        # Example: Fetch data from 2000 to 2022
        df_macro = loader.build_dataset(start_date="2000-01-01", end_date="2022-01-01")
        
        # Add a 12-month forward recession target
        df_final = loader.add_forward_target(df_macro, horizon=12)
        
        print("Data Shape:", df_final.shape)
        print("\nFirst 5 rows:")
        print(df_final.head())
        print("\nColumns:", df_final.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")