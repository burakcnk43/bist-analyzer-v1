import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from sklearn.cluster import KMeans

class FailurePatternMinerV3:
    """
    Analyzes historical group failures and clusters them into 'Trap States'.
    Generates dynamic veto rules for the optimizer.
    """
    def __init__(self, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_failure_rates = {}
        self.is_fitted = False

    def fit_trap_clusters(self, X: pd.DataFrame, hit_counts: pd.Series):
        """
        Clusters market environments where Top5_3PLUS failed.
        """
        X_clean = X.fillna(0)
        self.kmeans.fit(X_clean)

        clusters = self.kmeans.predict(X_clean)
        df = pd.DataFrame({'cluster': clusters, 'hit_count': hit_counts})

        # Calculate failure rate per cluster (3PLUS failure)
        self.cluster_failure_rates = df.groupby('cluster')['hit_count'].apply(lambda x: (x < 3).mean()).to_dict()
        self.is_fitted = True
        logger.info(f"Fitted {len(self.cluster_failure_rates)} trap clusters.")

    def get_trap_probability(self, X: pd.DataFrame) -> float:
        """
        Returns probability of being in a high-failure cluster.
        """
        if not self.is_fitted: return 0.0

        cluster = self.kmeans.predict(X.fillna(0))[0]
        return self.cluster_failure_rates.get(cluster, 0.5)

class GroupOutcomePredictorV2:
    """
    Predicts the probability of achieving various hit counts for a given Top-K set.
    Multi-target model: P(Hits >= 1), P(Hits >= 2), P(Hits >= 3), P(Hits >= 4), P(Hits >= 5).
    """
    def __init__(self):
        # We use a multi-output regressor or separate models.
        # For XGBoost, we'll use one model per target for better precision.
        self.models = {k: xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
                       for k in [1, 2, 3, 4, 5]}
        self.feature_cols = []

    def prepare_group_features(self,
                               market_row: pd.Series,
                               candidate_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates market state and pool-level statistics.
        """
        features = {}

        # 1. Market State
        features['mkt_rsi'] = market_row.get('market_z_rsi_14', 0)
        features['mkt_vol'] = market_row.get('market_z_rolling_std_20', 0)
        features['mkt_breadth'] = market_row.get('pct_above_sma50', 0.5)
        features['breadth_mom'] = market_row.get('breadth_momentum', 0)
        features['mkt_box_density'] = market_row.get('pct_box_breakout_up', 0)

        # 2. Pool Statistics
        if not candidate_pool.empty:
            alphas = candidate_pool['production_alpha']
            features['pool_avg_alpha'] = alphas.mean()
            features['pool_std_alpha'] = alphas.std()
            features['pool_min_alpha'] = alphas.min()
            features['pool_entropy'] = -np.sum(alphas * np.log(alphas + 1e-9)) # Uncertainty proxy

            # Diversity
            features['unique_sectors'] = candidate_pool['sector'].nunique()

            # Correlation density
            if 'deception_score' in candidate_pool.columns:
                features['avg_failure_risk'] = candidate_pool['deception_score'].mean()
            else:
                features['avg_failure_risk'] = 0.0
        else:
            for c in ['pool_avg_alpha', 'pool_std_alpha', 'pool_min_alpha', 'pool_entropy', 'unique_sectors', 'avg_failure_risk']:
                features[c] = 0

        return pd.DataFrame([features])

    def train(self, X: pd.DataFrame, hit_counts: pd.Series):
        """
        Trains models for each threshold.
        """
        self.feature_cols = [c for c in X.columns if not c.startswith('failed_') and c != 'hit_count']
        X_clean = X[self.feature_cols].fillna(0)

        logger.info(f"Training GroupOutcomePredictorV2 on {len(X)} days...")
        for k, model in self.models.items():
            y = (hit_counts >= k).astype(int)
            if y.nunique() > 1:
                model.fit(X_clean, y)
            else:
                logger.warning(f"Target Hits >= {k} has only one class. Skipping.")

    def predict_hit_distribution(self, group_X: pd.DataFrame) -> Dict[int, float]:
        """
        Returns probabilities for Hits >= 1..5.
        """
        if not self.feature_cols:
            return {k: 0.5 for k in [1, 2, 3, 4, 5]}

        X = group_X[self.feature_cols].fillna(0)
        probs = {}
        for k, model in self.models.items():
            try:
                probs[k] = float(model.predict_proba(X)[:, 1][0])
            except:
                probs[k] = 0.0 # If not trained
        return probs
