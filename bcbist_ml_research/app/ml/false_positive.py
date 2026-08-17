import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class FalsePositivePredictor:
    """
    Specifically targets candidates that look strong but have a high probability
    of failing (False Positives).

    Trained on historical "Mistakes": Top-rank candidates that resulted in 5D losses.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=2.0 # Bias towards detecting failures
        )
        self.feature_cols = []

    def train(self, X: pd.DataFrame, is_top_candidate: pd.Series, is_failure: pd.Series):
        """
        X: Combined stock and market features.
        is_top_candidate: Boolean mask for stocks that were highly ranked by base models.
        is_failure: Boolean mask for stocks that had negative returns.
        """
        # We only train on candidates that the base model would have picked
        train_mask = is_top_candidate
        X_train = X[train_mask].fillna(0)
        y_train = is_failure[train_mask].astype(int)

        if len(X_train) < 100:
            logger.warning("Insufficient samples for FalsePositivePredictor training.")
            return

        self.feature_cols = [c for c in X_train.columns if c not in ['date', 'symbol', 'target']]
        X_train_clean = X_train[self.feature_cols]

        logger.info(f"Training FalsePositivePredictor on {len(X_train_clean)} top candidates...")
        self.model.fit(X_train_clean, y_train)

    def predict_failure_prob(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability that the candidate is a False Positive.
        """
        if not self.feature_cols:
            return np.zeros(len(X))

        available = [c for c in self.feature_cols if c in X.columns]
        X_clean = X[available]

        if len(available) < len(self.feature_cols):
            X_clean = X_clean.reindex(columns=self.feature_cols, fill_value=0)

        return self.model.predict_proba(X_clean)[:, 1]
