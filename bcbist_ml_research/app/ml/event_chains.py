import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EventChainExpert:
    """
    Models the causal chain: Event -> Sector Response -> Stock Relative Strength.
    Tracks 'Impulse Response' for specific KAP event types.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []

    def prepare_event_features(self,
                               symbol_df: pd.DataFrame,
                               event_df: pd.DataFrame,
                               market_row: pd.Series) -> pd.DataFrame:
        """
        Features:
        - Event Direction (KAP sentiment)
        - Time since last event
        - Sector momentum post-event
        - Symbol relative strength acceleration
        """
        df = pd.DataFrame(index=symbol_df.index)

        # Merge event flags
        df = df.join(event_df[['event_direction', 'event_importance', 'time_since_event']], how='left')

        # Sector Context
        df['sector_mom_5d'] = symbol_df['sector_mean_return_5d']

        # Interaction: Event x Market Regime
        regime_map = {"BEAR": -1, "CRASH": -2, "SIDEWAYS": 0, "BULL": 1}
        df['regime_val'] = regime_map.get(market_row.get('regime', 'NORMAL'), 0)

        # Impulse Response: High volume post-event?
        df['event_vol_spike'] = (symbol_df['relative_volume'] > 1.5).astype(int) * (df['time_since_event'] < 2).astype(int)

        return df.fillna(0)

    def train(self, X: pd.DataFrame, y_success: pd.Series):
        self.feature_cols = X.columns.tolist()
        logger.info(f"Training EventChainExpert on {len(X)} instances...")
        self.model.fit(X, y_success.astype(int))

    def predict_event_impulse(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols:
            return np.full(len(X), 0.5)

        X_clean = X[self.feature_cols].fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]
