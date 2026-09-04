import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AlphaTrustModel:
    """
    Predicts P(BaseAlphaIsCorrect).
    Used to adjust confidence during high-risk market transitions or concept drift.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []

    def train(self, X: pd.DataFrame, is_success: pd.Series):
        """
        X: Context features (market state, model confidence, disagreement).
        is_success: 1 if individual stock had 5D return > 0.
        """
        # Automatically exclude non-numeric columns
        self.feature_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in ['symbol', 'date', 'target']]
        X_clean = X[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(f"Training AlphaTrustModel on {len(X_clean)} samples with {len(self.feature_cols)} features...")
        self.model.fit(X_clean, is_success.astype(int))

    def predict_trust_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns trust probability [0-1].
        """
        if not self.feature_cols:
            return np.full(len(X), 0.5)

        available = [c for c in self.feature_cols if c in X.columns]
        X_clean = X[available]
        if len(available) < len(self.feature_cols):
             X_clean = X_clean.reindex(columns=self.feature_cols, fill_value=0)

        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]

    def save(self, path: Path):
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols
        }, path)
        logger.info(f"AlphaTrustModel saved to {path}")

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.feature_cols = data['feature_cols']
        return obj
