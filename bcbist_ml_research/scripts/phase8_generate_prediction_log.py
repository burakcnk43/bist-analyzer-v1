import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictionLogGen")

def main():
    logger.info("--- PHASE 8: GENERATING HISTORICAL PREDICTION LOG ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            df['sector_col'] = entry.get('sector', 'Other')
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    # Define Research range (avoid holdout for log training data, but let's log everything)
    # We want a log that the "Meta Model" can learn from.
    # Usually we use cross-validation predictions for this.

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col', 'symbol', 'sector']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # Use Walk-Forward to get OOS predictions for the research period
    # n_folds=5 to get more data for mining
    from app.validation.walk_forward import WalkForwardValidation
    wf = WalkForwardValidation(n_folds=5, train_size_days=252*2, test_size_days=120, purge_days=20)

    # We'll log results here
    prediction_log = []

    splits = wf.split(global_df)

    for fold, (train_df, test_df) in enumerate(splits):
        logger.info(f"Processing Fold {fold+1}...")

        # Train P7 Ensemble
        p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
        tr_c = train_df[selected_features + [target_cls]].dropna()
        p_cls.fit(tr_c[selected_features], tr_c[target_cls])

        p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
        tr_r = train_df[selected_features + [target_rnk]].dropna()
        p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

        ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

        # Predict on Test Set (OOS)
        test_data = test_df.dropna(subset=[target_ret]).copy()
        if test_data.empty: continue

        scores = ensemble.get_alpha_score(test_data[selected_features])
        test_data['alpha_score'] = scores

        # Identify Ranks
        test_data['daily_rank'] = test_data.groupby(level=0)['alpha_score'].rank(ascending=False, method='first')

        # Collect info
        # To save space, we only log Top 20 stocks per day, plus some random samples if needed
        # Or just log everything if it's not too huge. 495 * 120 * 5 = ~300k rows.
        # Let's log Top 50 to keep it manageable.
        top_n_day = test_data[test_data['daily_rank'] <= 50].copy()

        # Add success labels
        top_n_day['is_hit'] = (top_n_day[target_ret] > 0).astype(int)
        top_n_day['is_top5'] = (top_n_day['daily_rank'] <= 5).astype(int)

        prediction_log.append(top_n_day)

    final_log = pd.concat(prediction_log)
    path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    final_log.to_csv(path)

    logger.info(f"Prediction log saved to {path} ({len(final_log)} rows)")

if __name__ == "__main__":
    main()
