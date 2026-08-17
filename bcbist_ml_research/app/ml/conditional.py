import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ConditionalStockScorer:
    """
    Calculates P(Stock succeeds | Market Environment).
    Distinguishes between general individual probability and the probability
    of being part of a winning group.
    """
    def __init__(self):
        # We model the interaction between stock features and market context
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        self.stock_features = []
        self.market_features = []

    def train(self, X: pd.DataFrame, y_hit: pd.Series, y_day_success: pd.Series):
        """
        X: combined stock and market features.
        y_hit: stock 5D return > 0.
        y_day_success: Top5_3PLUS occurred on that day.
        """
        # Target: Stock hit AND it was a successful day
        y_combined = (y_hit & y_day_success).astype(int)

        logger.info(f"Training ConditionalStockScorer on {len(X)} samples...")
        self.model.fit(X, y_combined)
        self.feature_cols = X.columns.tolist()

    def predict_conditional_prob(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns P(hit | market_state).
        """
        # Ensure only numeric columns are used and match training cols
        available = [c for c in self.feature_cols if c in X.columns]
        X_clean = X[available].fillna(0)

        # If some columns are missing, reindex with 0s
        if len(available) < len(self.feature_cols):
             X_clean = X.reindex(columns=self.feature_cols, fill_value=0)

        return self.model.predict_proba(X_clean)[:, 1]
