import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.metrics import accuracy_score
from app.ml.classification import DirectionalModel
from app.validation.walk_forward import WalkForwardValidation, purge_overlap

logger = logging.getLogger(__name__)

class AblationTester:
    def __init__(self, global_df: pd.DataFrame, target_col='target_up_5d'):
        self.df = global_df
        self.target = target_col
        self.wf = WalkForwardValidation(n_folds=2, train_size_days=252, test_size_days=60, purge_days=20)

    def get_feature_sets(self) -> Dict[str, List[str]]:
        cols = self.df.columns
        tech = [c for c in cols if any(x in c for x in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'adx'])]
        vol = [c for c in cols if any(x in c for x in ['volume', 'obv', 'pv_corr'])]
        sector = [c for c in cols if 'sector_' in c or 'relative_' in c]
        macro = [c for c in cols if 'macro_' in c]
        fund = [c for c in cols if any(x in c for x in ['pe_', 'pb_', 'roe', 'roa', 'margin'])]
        macro_events = [c for c in cols if 'event_' in c and 'company' not in c]
        company_kap = [c for c in cols if 'event_' in c or 'days_since' in c]

        def clean(feats):
            return sorted(list(set([f for f in feats if f in cols])))

        return {
            "A: Technical": clean(tech),
            "B: Tech + Vol": clean(tech + vol),
            "C: + Sector": clean(tech + vol + sector),
            "D: + Macro": clean(tech + vol + sector + macro),
            "E: + Fundamental": clean(tech + vol + sector + macro + fund),
            "F: + Macro Events": clean(tech + vol + sector + macro + fund + macro_events),
            "G: FULL (Inc. Company KAP)": clean(tech + vol + sector + macro + fund + company_kap)
        }

    def run_tests(self) -> pd.DataFrame:
        feature_sets = self.get_feature_sets()
        results = []

        splits = self.wf.split(self.df)

        for name, features in feature_sets.items():
            if not features:
                logger.warning(f"Feature set {name} is empty. Skipping.")
                continue

            logger.info(f"Running Ablation Test: {name} ({len(features)} features)")

            fold_accs = []
            for fold, (train_df, test_df) in enumerate(splits):
                train_clean = train_df.dropna(subset=[self.target])
                test_clean = test_df.dropna(subset=[self.target])

                if train_clean.empty or test_clean.empty:
                    continue

                X_train = train_clean[features].replace([np.inf, -np.inf], 0).fillna(0)
                y_train = train_clean[self.target]

                X_test = test_clean[features].replace([np.inf, -np.inf], 0).fillna(0)
                y_test = test_clean[self.target]

                model = DirectionalModel(model_type="xgboost")
                model.train(X_train, y_train)

                y_pred = model.predict(X_test)
                fold_accs.append(accuracy_score(y_test, y_pred))

            results.append({
                "Configuration": name,
                "Feature Count": len(features),
                "Avg Accuracy": np.mean(fold_accs) if fold_accs else 0
            })

        return pd.DataFrame(results)
