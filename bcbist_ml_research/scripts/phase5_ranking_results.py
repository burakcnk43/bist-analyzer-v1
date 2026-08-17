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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RankingResults")

def main():
    logger.info("--- PHASE 5: CROSS-SECTIONAL RANKING AUDIT ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 20
    target_cls = f'target_up_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # Train Global Model
    pipeline = ResearchPipeline(model_type='xgboost', n_features=30)
    train_clean = research_df[selected_features + [target_cls]].dropna()
    pipeline.fit(train_clean[selected_features], train_clean[target_cls])

    # Ranking evaluation
    test_data = holdout_df[selected_features + [target_cls, target_ret]].dropna()
    test_data['prob'] = pipeline.predict_proba(test_data[selected_features])

    # Daily Ranking
    # For each day, take top 5 and bottom 5 by probability
    def get_group_returns(group_df):
        top_5 = group_df.nlargest(5, 'prob')
        bot_5 = group_df.nsmallest(5, 'prob')
        return pd.Series({
            'top_5_ret': top_5[target_ret].mean(),
            'bot_5_ret': bot_5[target_ret].mean(),
            'spread': top_5[target_ret].mean() - bot_5[target_ret].mean()
        })

    daily_results = test_data.groupby(level=0).apply(get_group_returns)

    avg_results = daily_results.mean()

    print("\nRanking Audit Summary (Avg over holdout):")
    print(avg_results)

    daily_results.to_csv(DATA_REPORTS_DIR / "PHASE5_RANKING_RESULTS.csv")

if __name__ == "__main__":
    main()
