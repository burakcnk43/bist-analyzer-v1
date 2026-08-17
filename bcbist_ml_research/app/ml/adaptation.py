import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ReliabilityTracker:
    """
    Maintains rolling performance metrics for all specialist components.
    Used by Gating Model to adjust weights.
    """
    def __init__(self, windows=[10, 20, 60]):
        self.windows = windows
        self.history = pd.DataFrame()

    def add_feedback(self, model_name: str, date: pd.Timestamp, symbol: str, is_hit: bool):
        new_row = pd.DataFrame([{
            "model": model_name,
            "date": date,
            "symbol": symbol,
            "hit": int(is_hit)
        }])
        self.history = pd.concat([self.history, new_row]).drop_duplicates(subset=['model', 'date', 'symbol'])

    def get_model_trust(self, model_name: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        Returns precision scores across windows.
        """
        if self.history.empty: return {f"p_{w}": 0.5 for w in self.windows}

        m_hist = self.history[(self.history['model'] == model_name) & (self.history['date'] < date)]
        if m_hist.empty: return {f"p_{w}": 0.5 for w in self.windows}

        trust = {}
        for w in self.windows:
            recent = m_hist.tail(w)
            trust[f"p_{w}"] = recent['hit'].mean()

        return trust

class FeatureRelationAdapter:
    """
    Detects if feature-return correlations are drifting in the recent window.
    """
    def __init__(self, window=20):
        self.window = window

    def get_recent_correlations(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        recent = df.tail(self.window)
        return recent.corr()[target_col].drop(target_col)
