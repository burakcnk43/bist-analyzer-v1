import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AdaptiveMetaLearnerV4:
    """
    Dynamic Meta-Learner that adjusts expert weights based on rolling OOS performance.
    Tracks: General, Sector, Regime, Darvas, and AlphaTrust specialists.
    """
    def __init__(self, windows=[10, 20, 60]):
        self.windows = windows
        self.specialist_history = {} # name -> list of outcomes
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        self.is_trained = False
        self.feature_cols = []

    def record_outcome(self, specialist_name: str, date: pd.Timestamp, is_success: bool):
        if specialist_name not in self.specialist_history:
            self.specialist_history[specialist_name] = []
        self.specialist_history[specialist_name].append({'date': date, 'hit': int(is_success)})

    def get_reliability_features(self, current_date: pd.Timestamp) -> Dict[str, float]:
        features = {}
        for name, history in self.specialist_history.items():
            df = pd.DataFrame(history)
            if df.empty:
                for w in self.windows: features[f'rel_{name}_{w}d'] = 0.5
                continue
            recent = df[df['date'] < current_date]
            for w in self.windows:
                if len(recent) >= w:
                    features[f'rel_{name}_{w}d'] = recent.tail(w)['hit'].mean()
                else:
                    features[f'rel_{name}_{w}d'] = 0.5 # Neutral
        return features

    def prepare_meta_X(self,
                       specialist_probs: Dict[str, np.ndarray],
                       market_quality: float,
                       transition_prob: float,
                       reliability_feats: Dict[str, float]) -> pd.DataFrame:
        """
        Creates a row per candidate with expert predictions + context.
        """
        meta_df = pd.DataFrame(specialist_probs)
        meta_df['market_quality'] = market_quality
        meta_df['transition_risk'] = transition_prob
        for k, v in reliability_feats.items():
            meta_df[k] = v
        return meta_df.fillna(0.5)

    def train(self, X: pd.DataFrame, y_success: pd.Series):
        self.feature_cols = X.columns.tolist()
        logger.info(f"Training AdaptiveMetaLearnerV4 on {len(X)} samples...")
        self.model.fit(X, y_success)
        self.is_trained = True

    def predict_alpha(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            expert_cols = [c for c in X.columns if c.startswith('p_') or c in ['general', 'sector', 'regime', 'box']]
            return X[expert_cols].mean(axis=1).values
        return self.model.predict(X[self.feature_cols])

class AdaptiveMetaLearnerV5(AdaptiveMetaLearnerV4):
    """
    Advanced Meta-Learner with Contextual Reliability and Exponential Decay.
    """
    def __init__(self, windows=[10, 20, 60, 120], decay_factor=0.98):
        super().__init__(windows)
        self.decay_factor = decay_factor

    def get_contextual_reliability(self, current_date: pd.Timestamp, regime: str) -> Dict[str, float]:
        features = {}
        for name, history in self.specialist_history.items():
            df = pd.DataFrame(history)
            recent = df[df['date'] < current_date].copy()
            if recent.empty:
                for w in self.windows: features[f'rel_{name}_{w}d'] = 0.5
                continue
            recent['days_ago'] = (current_date - recent['date']).dt.days
            recent['weight'] = self.decay_factor ** recent['days_ago']
            if 'regime' in recent.columns:
                recent.loc[recent['regime'] == regime, 'weight'] *= 1.5
            for w in self.windows:
                w_subset = recent.tail(w)
                w_sum = (w_subset['hit'] * w_subset['weight']).sum()
                w_total = w_subset['weight'].sum() + 1e-9
                features[f'rel_{name}_{w}d'] = w_sum / w_total
        return features

    def train_v5(self, X: pd.DataFrame, y_success: pd.Series):
        self.feature_cols = X.columns.tolist()
        logger.info(f"Training AdaptiveMetaLearnerV5 on {len(X)} instances...")
        self.model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03)
        self.model.fit(X, y_success)
        self.is_trained = True

    def save_state(self, path: Path):
        joblib.dump({
            'history': self.specialist_history,
            'model': self.model,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }, path)

    def load_state(self, path: Path):
        if path.exists():
            data = joblib.load(path)
            self.specialist_history = data.get('history', {})
            self.model = data.get('model')
            self.feature_cols = data.get('feature_cols', [])
            self.is_trained = data.get('is_trained', False)
