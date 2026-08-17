import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional
from app.ml.pipeline import ResearchPipeline

logger = logging.getLogger(__name__)

class BoxExpert:
    """
    XGBoost specialist focused on Darvas Box structures and breakouts.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.box_features = [
            'box_height_pct', 'dist_box_top', 'pos_in_box', 'box_tightness',
            'box_compression', 'is_breakout_close', 'breakout_strength',
            'breakout_volume_ratio', 'box_age', 'box_staircase_score'
        ]

    def train(self, df: pd.DataFrame, target_col: str):
        # Filter rows with box features
        available = [f for f in self.box_features if f in df.columns]
        train_df = df.dropna(subset=available + [target_col])

        logger.info(f"Training BoxExpert on {len(train_df)} samples...")
        self.model.fit(train_df[available], train_df[target_col])

    def predict_probs(self, X: pd.DataFrame) -> np.ndarray:
        available = [f for f in self.box_features if f in X.columns]
        return self.model.predict_proba(X[available])[:, 1]

class BoxFailurePredictor:
    """
    Predicts the probability of a breakout failing (False Breakout / Bull Trap).
    Target: breakout happened but return_5d < 0.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=13)
        self.features = ['breakout_strength', 'breakout_volume_ratio', 'rsi_14', 'bb_width', 'box_age']

    def train(self, df: pd.DataFrame, ret_5d_col: str):
        # Filter for breakout days
        breakouts = df[df['is_breakout_close'] == 1].copy()
        if len(breakouts) < 100:
            logger.warning("Insufficient breakouts to train BoxFailurePredictor.")
            return

        y = (breakouts[ret_5d_col] < 0).astype(int)
        available = [f for f in self.features if f in breakouts.columns]

        logger.info(f"Training BoxFailurePredictor on {len(breakouts)} breakouts...")
        self.model.fit(breakouts[available].fillna(0), y)

    def predict_failure_prob(self, X: pd.DataFrame) -> np.ndarray:
        available = [f for f in self.features if f in X.columns]
        return self.model.predict_proba(X[available].fillna(0))[:, 1]
