import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR, MODELS_DIR
from app.ml.ensemble import AlphaEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiLogGen")

def main():
    logger.info("--- PHASE 10: GENERATING MULTI-HORIZON OOS PREDICTION LOG ---")

    # 1. Load Data
    all_data_list = []
    # Use top 100 symbols for faster log generation in research
    for entry in STOCK_UNIVERSE[:100]:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    # Use MultiIndex for unique identification
    global_df = global_df.reset_index().set_index(['date', 'symbol_col'])

    from app.validation.walk_forward import WalkForwardValidation
    wf = WalkForwardValidation(n_folds=3, train_size_days=252*2, test_size_days=90, purge_days=20)

    horizons = [1, 3, 5]
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    # Filter features from multiindex columns if necessary, but here we just need names
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    all_logs = []

    # WalkForwardValidation split logic works on single index typically,
    # but since it uses loc[dates], it should work if we pass it the index level 0 unique dates.
    # Actually, let's keep it simple:
    unique_dates = global_df.index.get_level_values(0).unique()

    # Manually split to ensure PIT
    for fold in range(wf.n_folds):
        logger.info(f"Processing Fold {fold+1}...")

        test_end = unique_dates[-1] - timedelta(days=fold * wf.test_size_days)
        test_start = test_end - timedelta(days=wf.test_size_days)
        train_end = test_start - timedelta(days=wf.purge_days)
        train_start = train_end - timedelta(days=wf.train_size_days)

        train_df = global_df.loc[train_start:train_end]
        test_df = global_df.loc[test_start:test_end]

        if train_df.empty or test_df.empty: continue

        fold_predictions = pd.DataFrame(index=test_df.index)

        for h in horizons:
            logger.info(f"  Horizon {h}d...")
            target_cls = f'target_up_{h}d'
            target_rnk = f'target_label_{h}d'
            target_ret = f'target_return_{h}d'

            from app.ml.pipeline import ResearchPipeline
            p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
            tr_c = train_df[selected_features + [target_cls]].dropna()
            p_cls.fit(tr_c[selected_features], tr_c[target_cls])

            p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
            tr_r = train_df[selected_features + [target_rnk]].dropna()
            p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

            ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

            # Predict
            test_subset = test_df.dropna(subset=[target_ret]).copy()
            if test_subset.empty: continue

            scores = ensemble.get_alpha_score(test_subset[selected_features])
            fold_predictions.loc[test_subset.index, f'score_{h}d'] = scores
            fold_predictions.loc[test_subset.index, f'actual_{h}d'] = test_subset[target_ret]

        all_logs.append(fold_predictions.dropna())

    final_log = pd.concat(all_logs)
    path = DATA_REPORTS_DIR / "PHASE10_MULTI_HORIZON_LOG.csv"
    final_log.to_csv(path)
    logger.info(f"Multi-horizon log saved to {path} ({len(final_log)} rows)")

if __name__ == "__main__":
    main()
