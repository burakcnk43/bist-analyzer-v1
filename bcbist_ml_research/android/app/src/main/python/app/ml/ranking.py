import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class RankingModel:
    def __init__(self, model_type="xgboost", params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.features = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains a ranking model.
        """
        self.features = X.columns.tolist()

        # We assume X and y are already aligned row-by-row by the pipeline/caller.
        # We only need to ensure they are sorted by date so groups are contiguous for XGBRanker.
        # To avoid the .loc index explosion (due to duplicate dates), we use a temporary column.

        tmp = X.copy()
        tmp['_y_target'] = y.values
        tmp = tmp.sort_index()

        X_sorted = tmp.drop(columns=['_y_target'])
        y_sorted = tmp['_y_target']

        # qid: query id per date
        qids, _ = pd.factorize(X_sorted.index)

        if self.model_type == "xgboost":
            default_params = {
                "objective": "rank:ndcg",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "random_state": 42
            }
            self.model = xgb.XGBRanker(**{**default_params, **self.params})

            # Use values to avoid any index alignment logic inside XGBoost
            self.model.fit(X_sorted.values, y_sorted.values, qid=qids)
        else:
            raise ValueError(f"Ranking model {self.model_type} not supported.")

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        # Note: XGBRanker predict doesn't care about groups
        return self.model.predict(X[self.features].values)
