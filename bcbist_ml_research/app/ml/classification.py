import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

class DirectionalModel:
    def __init__(self, model_type="xgboost", params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.features = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.features = X.columns.tolist()

        if self.model_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
            self.model = xgb.XGBClassifier(**{**default_params, **self.params})
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000)

        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def get_importance(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self.features)
        elif hasattr(self.model, "coef_"):
            return pd.Series(np.abs(self.model.coef_[0]), index=self.features)
        return pd.Series()
