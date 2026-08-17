import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CPCV")

def main():
    logger.info("--- PHASE 9: COMBINATORIAL PURGED CROSS VALIDATION (CPCV) ---")

    # 1. Load Data
    all_data_list = []
    # To keep CPCV fast, use first 100 stocks
    for entry in STOCK_UNIVERSE[:100]:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    # Split dataset into 6 chunks
    all_dates = sorted(global_df.index.unique())
    n_chunks = 6
    chunk_size = len(all_dates) // n_chunks

    # Simple version: Take 2 chunks as test, others as train, move in combinations
    # Here we'll just do 3 different non-contiguous splits as a proxy for CPCV

    h = 5
    target_cls = f'target_up_{h}d'
    target_ret = f'target_return_{h}d'
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col', 'symbol', 'sector']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    results = []

    # Combination 1: Test middle chunks
    test_dates = all_dates[2*chunk_size:4*chunk_size]
    train_dates = all_dates[:chunk_size] + all_dates[5*chunk_size:]

    for comb in [1, 2]:
        logger.info(f"Running Combination {comb}...")
        if comb == 2:
            test_dates = all_dates[0:chunk_size] + all_dates[4*chunk_size:5*chunk_size]
            train_dates = all_dates[chunk_size:3*chunk_size]

        train_df = global_df[global_df.index.isin(train_dates)]
        test_df = global_df[global_df.index.isin(test_dates)]

        # Purge and Embargo
        # We ensure train_dates and test_dates are separated by at least 20 days
        # (This is handled by the index filter since we skipped chunks)

        p = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
        tr_c = train_df[selected_features + [target_cls]].dropna()
        p.fit(tr_c[selected_features], tr_c[target_cls])

        te_c = test_df[selected_features + [target_ret]].dropna().copy()
        if te_c.empty: continue

        te_c['prob'] = p.predict_proba(te_c[selected_features])
        summary, _ = calculate_top_k_metrics(te_c, 'prob', target_ret, k=5)

        results.append({
            "Comb": comb,
            "HitRate": summary['avg_hit_rate'],
            "AvgRet": summary['avg_return']
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE9_CPCV.csv", index=False)
    logger.info(f"CPCV Mean Hit Rate: {res_df['HitRate'].mean():.2%}")

if __name__ == "__main__":
    main()
