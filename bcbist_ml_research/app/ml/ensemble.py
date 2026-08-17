import numpy as np
import pandas as pd
from typing import List
from app.ml.pipeline import ResearchPipeline

class AlphaEnsemble:
    """
    Combines Classification and Ranking outputs into a single Alpha Score.
    Uses Daily Normalization to ensure stable cross-sectional ranking.
    """
    def __init__(self, classifier: ResearchPipeline, ranker: ResearchPipeline, w_cls=0.5, w_rnk=0.5):
        self.classifier = classifier
        self.ranker = ranker
        self.w_cls = w_cls
        self.w_rnk = w_rnk
        self.performance_history = []

    def set_weights(self, w_cls: float, w_rnk: float):
        self.w_cls = w_cls
        self.w_rnk = w_rnk

    def get_alpha_score(self, X: pd.DataFrame) -> np.ndarray:
        # 1. Classification Probabilities (already 0-1)
        cls_probs = self.classifier.predict_proba(X)

        # 2. Ranking Scores
        rnk_scores = self.ranker.predict(X)

        # 3. Daily Normalization (Z-score per date)
        # We handle both single-day and multi-day batches
        temp_df = pd.DataFrame({'rnk': rnk_scores}, index=X.index)

        def normalize_group(g):
            if len(g) > 1 and g.std() > 1e-9:
                return (g - g.mean()) / g.std()
            return g - g.mean()

        # If MultiIndex (date, symbol), group by level 0
        if isinstance(temp_df.index, pd.MultiIndex):
            rnk_norm = temp_df.groupby(level=0)['rnk'].transform(normalize_group)
        else:
            # If single Index, check if it has duplicates (multiple symbols per date)
            if temp_df.index.duplicated().any():
                rnk_norm = temp_df.groupby(temp_df.index)['rnk'].transform(normalize_group)
            else:
                # Single day or no duplicates
                rnk_norm = normalize_group(temp_df['rnk'])

        # 4. Map to 0-1 using Sigmoid
        rnk_probs = 1 / (1 + np.exp(-rnk_norm.values))

        return (self.w_cls * cls_probs) + (self.w_rnk * rnk_probs)
