import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional
import joblib

logger = logging.getLogger(__name__)

class DailyHitRatePredictor:
    """
    Predicts the probability that a given trading day will result in
    at least 3 out of 5 positive selections (Top5_3PLUS).
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.feature_cols = []

    def train(self, market_features: pd.DataFrame, hit_labels: pd.Series):
        """
        market_features: daily breadth, sector, macro, regime features.
        hit_labels: 1 if Top5 on that day had >=3 hits, 0 otherwise.
        """
        self.feature_cols = market_features.columns.tolist()
        logger.info(f"Training DailyHitRatePredictor on {len(market_features)} days...")
        self.model.fit(market_features, hit_labels)

    def predict_hit_prob(self, market_features: pd.DataFrame) -> float:
        """
        Returns P(3PLUS) for the day.
        """
        if self.model is None or not self.feature_cols:
            return 0.5

        X = market_features[self.feature_cols].tail(1).fillna(0)
        if X.empty: return 0.5

        return float(self.model.predict_proba(X)[:, 1][0])

class DailyConfidenceGate:
    """
    Determines selection mode (HIGH_CONFIDENCE, NORMAL, DEFENSIVE)
    based on predicted hit rates and drift/OOD metrics.
    """
    def __init__(self, high_threshold=0.6, defensive_threshold=0.35):
        self.high_threshold = high_threshold
        self.defensive_threshold = defensive_threshold

    def get_selection_mode(self,
                           p_3plus: float,
                           drift_score: float = 0.0,
                           ood_score: float = 0.0) -> str:
        """
        Logic for adaptive modes.
        """
        # Penalize probability with drift/OOD
        confidence = p_3plus * (1.0 - drift_score) * (1.0 - ood_score)

        if confidence >= self.high_threshold:
            return "HIGH_CONFIDENCE"
        if confidence <= self.defensive_threshold:
            return "DEFENSIVE"

        return "NORMAL"
