import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MetaModelV2:
    """
    Legacy Meta-Learner for Phase 11.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.feature_cols = []

    def prepare_meta_features(self,
                              base_X: pd.DataFrame,
                              scores_dict: Dict[str, np.ndarray],
                              regime: str,
                              similarity_stats: pd.DataFrame) -> pd.DataFrame:
        meta_df = pd.DataFrame(index=base_X.index)
        for name, scores in scores_dict.items():
            meta_df[f'sig_{name}'] = scores
        norm_signals = []
        for name in scores_dict:
            s = scores_dict[name]
            norm_signals.append((s - s.mean()) / (s.std() + 1e-9))
        meta_df['model_disagreement'] = np.std(norm_signals, axis=0)
        regime_map = {"BEAR": -1, "CRASH": -2, "SIDEWAYS": 0, "BULL": 1, "BULL_OVEREXTENDED": 2}
        meta_df['regime_val'] = regime_map.get(regime, 0)
        if 'rsi_14' in base_X.columns: meta_df['rsi_14'] = base_X['rsi_14']
        if 'market_z_rsi_14' in base_X.columns: meta_df['mkt_rsi'] = base_X['market_z_rsi_14']
        return meta_df.fillna(0)

    def predict_success_prob(self, meta_X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(meta_X[self.feature_cols])[:, 1]

class MetaGaterV3:
    """
    Advanced Meta-Learner for Phase 12.
    Gates specialists based on candidates, recent reliability, and transition probability.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        self.feature_cols = []

    def prepare_gating_features(self,
                                base_X: pd.DataFrame,
                                specialist_probs: Dict[str, np.ndarray],
                                regime: str,
                                transition_prob: float,
                                reliability_stats: Dict[str, float],
                                box_metrics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        meta_df = pd.DataFrame(index=base_X.index)

        # 1. Specialist Proposals
        for name, probs in specialist_probs.items():
            meta_df[f'p_{name}'] = probs

        # 2. Agreement/Disagreement
        prob_array = np.array(list(specialist_probs.values()))
        meta_df['proposal_std'] = np.std(prob_array, axis=0)
        meta_df['proposal_mean'] = np.mean(prob_array, axis=0)

        # 3. Market Context
        meta_df['transition_risk'] = transition_prob
        regime_map = {"BEAR": -1, "CRASH": -2, "SIDEWAYS": 0, "BULL": 1, "BULL_OVEREXTENDED": 2}
        meta_df['regime_val'] = regime_map.get(regime, 0)

        # 4. Reliability Context
        for name, trust in reliability_stats.items():
            meta_df[f'trust_{name}'] = trust

        # 5. Box Specifics (Phase 13)
        if box_metrics is not None:
            for col in box_metrics.columns:
                meta_df[f'box_{col}'] = box_metrics[col]

        # 6. Asset Specifics
        if 'rsi_14' in base_X.columns: meta_df['asset_rsi'] = base_X['rsi_14']

        return meta_df.fillna(0.5)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = X.columns.tolist()
        logger.info(f"Training MetaGaterV3 on {len(X)} samples...")
        self.model.fit(X, y)

    def predict_alpha(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

class AdaptiveEnsembleOptimizer:
    """
    Dynamically adjusts ensemble weights based on rolling OOS performance.
    """
    def __init__(self, window=60):
        self.window = window
        self.component_performance = {} # name -> rolling precision

    def update_performance(self, name: str, date: pd.Timestamp, is_hit: bool):
        if name not in self.component_performance:
            self.component_performance[name] = []
        self.component_performance[name].append({"date": date, "hit": int(is_hit)})

    def get_optimal_weights(self, current_date: pd.Timestamp) -> Dict[str, float]:
        weights = {}
        for name, history in self.component_performance.items():
            df = pd.DataFrame(history)
            recent = df[df['date'] < current_date].tail(self.window)
            if not recent.empty:
                weights[name] = recent['hit'].mean()
            else:
                weights[name] = 0.5 # Default

        # Normalize
        total = sum(weights.values())
        if total > 0:
            return {k: v/total for k, v in weights.items()}
        return {k: 1.0/len(weights) for k in weights}
