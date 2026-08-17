import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class SuccessDayClusterEngine:
    """
    Mines historical successful days (>= 3 hits) to find recurring signatures.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.centroids = None
        self.feature_cols = []

    def fit(self, market_features: pd.DataFrame, hit_counts: pd.Series):
        """
        Learns successful signatures.
        """
        # Only fit on successful days
        success_mask = hit_counts >= 3
        X_success = market_features[success_mask].fillna(0)

        if len(X_success) < self.n_clusters:
            logger.warning("Not enough successful days for clustering.")
            return

        self.feature_cols = market_features.columns.tolist()
        self.kmeans.fit(X_success)
        self.centroids = self.kmeans.cluster_centers_
        logger.info(f"Learned {self.n_clusters} success signatures from {len(X_success)} days.")

    def get_similarity_score(self, current_market: pd.DataFrame) -> float:
        """
        Returns a probability-like score [0-1] based on distance to nearest success centroid.
        """
        if self.centroids is None: return 0.5

        X = current_market[self.feature_cols].tail(1).fillna(0)
        if X.empty: return 0.5

        distances = cdist(X, self.centroids, metric='euclidean')
        min_dist = np.min(distances)

        # Simple RBF-like mapping: smaller distance -> higher score
        # Using 1 / (1 + dist) as a simple decay
        return 1.0 / (1.0 + min_dist)

class FailureDayClusterEngine:
    """
    Identifies "danger signatures" from days with 0 or 1 hits.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.centroids = None
        self.feature_cols = []

    def fit(self, market_features: pd.DataFrame, hit_counts: pd.Series):
        failure_mask = hit_counts <= 1
        X_fail = market_features[failure_mask].fillna(0)

        if len(X_fail) < self.n_clusters:
            return

        self.feature_cols = market_features.columns.tolist()
        self.kmeans.fit(X_fail)
        self.centroids = self.kmeans.cluster_centers_
        logger.info(f"Learned {self.n_clusters} failure signatures from {len(X_fail)} days.")

    def get_danger_score(self, current_market: pd.DataFrame) -> float:
        if self.centroids is None: return 0.0

        X = current_market[self.feature_cols].tail(1).fillna(0)
        if X.empty: return 0.0

        distances = cdist(X, self.centroids, metric='euclidean')
        min_dist = np.min(distances)

        return 1.0 / (1.0 + min_dist)
