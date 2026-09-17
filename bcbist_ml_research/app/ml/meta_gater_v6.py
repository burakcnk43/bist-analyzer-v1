import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class MetaGaterV6:
    """
    The 'Elite Brain' (Phase 28).
    Uses a hybrid approach:
    1. XGBoost for high-dimensional interaction mining.
    2. Random Forest for regime-robust stability.
    3. Consensus Gating for 80% accuracy objective.
    """
    def __init__(self):
        self.xgb = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.02,
            random_state=42,
            objective='binary:logistic'
        )
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        self.feature_cols = []
        self.is_trained = False

    def prepare_elite_features(self,
                               X: pd.DataFrame,
                               specialist_probs: Dict[str, np.ndarray],
                               regime: str,
                               mkt_quality: float) -> pd.DataFrame:
        meta_df = pd.DataFrame(specialist_probs, index=X.index)

        # 1. Market Health
        meta_df['mkt_quality'] = mkt_quality
        regime_map = {
            "BULL": 3, "RECOVERY": 2, "SIDEWAYS": 0,
            "SIDEWAYS_LOW_VOL": -1, "BEAR": -3, "CRASH": -5
        }
        meta_df['regime_val'] = regime_map.get(regime, 0)

        # 2. Synergy Features
        meta_df['gen_trust_sync'] = meta_df['general'] * meta_df.get('trust', 0.5)
        meta_df['box_event_sync'] = meta_df.get('box', 0.5) * meta_df.get('event', 0.5)

        # 3. Core Indicators (Raw)
        for col in ['rsi_14', 'relative_volume', 'dist_sma_20', 'pv_synergy']:
            if col in X.columns: meta_df[col] = X[col]

        # 4. Global Synergy
        meta_df['storm_intensity'] = meta_df['general'] * meta_df.get('trust', 0.5) * meta_df.get('box', 0.5)

        return meta_df.fillna(0)

    def train(self, X_meta: pd.DataFrame, y_hits: pd.Series):
        X_clean = X_meta.select_dtypes(include=[np.number]).fillna(0)
        self.feature_cols = X_clean.columns.tolist()

        logger.info(f"Training Elite Hybrid Brain on {len(X_clean)} samples...")
        self.xgb.fit(X_clean, y_hits)
        self.rf.fit(X_clean, y_hits)
        self.is_trained = True

    def predict_elite_confidence(self, X_meta: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            return X_meta['general'].values

        X_clean = X_meta[self.feature_cols].fillna(0)
        p_xgb = self.xgb.predict_proba(X_clean)[:, 1]
        p_rf = self.rf.predict_proba(X_clean)[:, 1]

        # Phase 28 Hyper-Consensus:
        # Only stocks endorsed by both XGBoost and RF get elite status.
        consensus = np.where((p_xgb > 0.70) & (p_rf > 0.70), (p_xgb + p_rf) / 2, 0.45)
        return consensus

    def save(self, path: Path):
        joblib.dump({
            'xgb': self.xgb,
            'rf': self.rf,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }, path)

    @classmethod
    def load(cls, path: Path):
        if not path.exists(): return cls()
        data = joblib.load(path)
        obj = cls()
        obj.xgb = data['xgb']
        obj.rf = data['rf']
        obj.feature_cols = data['feature_cols']
        obj.is_trained = data['is_trained']
        return obj
