import pandas as pd
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier, export_text
from typing import List, Dict

logger = logging.getLogger(__name__)

class PatternMiner:
    """
    Minining the prediction log to find high-probability success clusters.
    """
    def __init__(self, log_path: str):
        self.df = pd.read_csv(log_path, index_col='date', parse_dates=True)
        self.rules = []

    def clean_X(self, X: pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan)
        # Cap extreme values to prevent float32 overflow
        X = X.clip(lower=-1e9, upper=1e9)
        return X.fillna(0)

    def mine_success_patterns(self, min_samples=100):
        """
        Uses a decision tree to find interpretable rules for 'is_hit' == 1.
        """
        top_picks = self.df[self.df['daily_rank'] <= 10].copy()

        feature_cols = [c for c in top_picks.columns if not any(x in c for x in ['target', 'score', 'rank', 'hit', 'symbol', 'sector', 'col'])]

        X = self.clean_X(top_picks[feature_cols])
        y = top_picks['is_hit']

        # Train a shallow tree for interpretability
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_samples)
        tree.fit(X, y)

        tree_rules = export_text(tree, feature_names=feature_cols)
        self.rules.append(tree_rules)

        return tree_rules

    def analyze_sector_performance(self):
        """
        Calculates hit rate per sector for the Top 5 picks.
        """
        top5 = self.df[self.df['is_top5'] == 1]
        sector_stats = top5.groupby('sector_col').agg({
            'is_hit': ['mean', 'count'],
            'target_return_5d': 'mean'
        })
        sector_stats.columns = ['hit_rate', 'sample_count', 'avg_return']
        return sector_stats.sort_values('hit_rate', ascending=False)

    def analyze_feature_ranges(self, feature_name: str, bins=5):
        """
        Finds which quantiles of a feature lead to higher hit rates.
        """
        top5 = self.df[self.df['is_top5'] == 1].copy()
        if feature_name not in top5.columns:
            return pd.DataFrame()

        top5[feature_name] = top5[feature_name].replace([np.inf, -np.inf], np.nan).fillna(0)
        top5['bucket'] = pd.qcut(top5[feature_name], q=bins, duplicates='drop')
        bucket_stats = top5.groupby('bucket').agg({
            'is_hit': 'mean',
            'target_return_5d': 'mean',
            'symbol_col': 'count'
        })
        return bucket_stats
