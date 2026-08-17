import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EventChainExpert:
    """
    Models the causal chain: Event -> Sector Response -> Stock Relative Strength.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []

    def train(self, X: pd.DataFrame, y_success: pd.Series):
        # Automatically exclude non-numeric columns
        self.feature_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in ['symbol', 'date', 'target']]
        X_clean = X[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info(f"Training EventChainExpert on {len(X_clean)} instances with {len(self.feature_cols)} features...")
        self.model.fit(X_clean, y_success.astype(int))

    def predict_event_impulse(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols:
            return np.full(len(X), 0.5)
        X_clean = X.reindex(columns=self.feature_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]

    def save(self, path: Path):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, path)

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.feature_cols = data['feature_cols']
        return obj
