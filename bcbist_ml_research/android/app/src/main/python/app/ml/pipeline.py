import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from app.ml.classification import DirectionalModel
from app.ml.ranking import RankingModel

logger = logging.getLogger(__name__)

class ResearchPipeline:
    def __init__(self, model_type='xgboost', n_features=30, task='classification'):
        self.model_type = model_type
        self.n_features = n_features
        self.task = task
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif, k=n_features)

        if task == 'classification':
            self.model = DirectionalModel(model_type=model_type)
        elif task == 'ranking':
            self.model = RankingModel(model_type=model_type)

        self.selected_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scaling
        X_scaled = self.scaler.fit_transform(X_clean)

        # Feature Selection
        k = min(self.n_features, X_clean.shape[1])
        self.selector.set_params(k=k)

        # If ranking, we use the raw target for selection
        self.selector.fit(X_scaled, y)
        mask = self.selector.get_support()
        self.selected_features = X_clean.columns[mask].tolist()

        # Training
        X_final = X_clean[self.selected_features]
        self.model.train(X_final, y)

    def predict(self, X: pd.DataFrame):
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_final = X_clean[self.selected_features]
        if self.task == 'classification':
            return self.model.predict(X_final)
        else:
            return self.model.predict_score(X_final)

    def predict_proba(self, X: pd.DataFrame):
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_final = X_clean[self.selected_features]
        if self.task == 'classification':
            return self.model.predict_proba(X_final)
        else:
            return self.model.predict_score(X_final)
