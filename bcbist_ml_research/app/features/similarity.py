import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """
    Finds historically similar situations and uses their outcomes as features.
    """
    def __init__(self, k=10):
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.scaler = StandardScaler()
        self.train_outcomes = None
        # Use more stable features for similarity
        self.features = ['rsi_14', 'bb_pct', 'relative_volume']

    def fit(self, train_df: pd.DataFrame, target_col: str):
        """
        Builds the searchable historical database.
        """
        # Ensure features exist
        available = [f for f in self.features if f in train_df.columns]
        self.features = available

        X = train_df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.train_outcomes = train_df[target_col].values

        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Building Similarity Index with {len(X)} cases...")
        self.knn.fit(X_scaled)

    def get_similarity_feature(self, X_query: pd.DataFrame) -> np.ndarray:
        """
        For each row in query, find avg outcome of k most similar historical days.
        """
        X_clean = X_query[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        # Find neighbors
        distances, indices = self.knn.kneighbors(X_scaled)

        # Calculate avg outcome of neighbors
        neighbor_outcomes = self.train_outcomes[indices]
        return np.mean(neighbor_outcomes, axis=1)
