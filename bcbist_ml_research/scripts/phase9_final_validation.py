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
from app.ml.meta_models import SuccessPredictor, FailurePredictor
from app.features.similarity import SimilarityEngine
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase9FinalValidation")

def main():
    logger.info("--- PHASE 9: UNSEEN FORWARD VALIDATION ---")

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

    # Define STRICT UNSEEN FORWARD SET (Last 60 Days)
    max_date = global_df.index.max()
    unseen_start = max_date - timedelta(days=60)

    # Training Data (Research + partial holdout from P8, up to unseen_start)
    # But strictly separated by 20 days purge
    train_end = unseen_start - timedelta(days=20)

    train_df = global_df[global_df.index < train_end]
    unseen_test_df = global_df[global_df.index >= unseen_start]

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col', 'symbol', 'sector']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # 2. Train Full Suite
    logger.info("Training Full Alpha Suite (Phase 8 architecture)...")
    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    tr_c = train_df[selected_features + [target_cls]].dropna()
    p_cls.fit(tr_c[selected_features], tr_c[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    tr_r = train_df[selected_features + [target_rnk]].dropna()
    p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

    base_ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

    # Meta-Learning (Requires OOS prediction log for training)
    # For speed, we'll re-use the P8 log but in real life we'd re-generate up to unseen_start
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    log_df = pd.read_csv(log_path)
    # Strictly remove unseen dates from log if they leaked in
    log_df['date'] = pd.to_datetime(log_df['date'])
    log_df = log_df[log_df['date'] < train_end]

    success_mdl = SuccessPredictor()
    success_mdl.train(log_df, selected_features)

    sim_engine = SimilarityEngine(k=20)
    sim_engine.fit(train_df.dropna(subset=[target_ret]), target_ret)

    # 3. Evaluate on COMPLETELY UNSEEN FORWARD DATA
    logger.info(f"Evaluating on UNSEEN period: {unseen_start.date()} to {max_date.date()}")
    test_data = unseen_test_df[selected_features + [target_ret]].dropna().copy()

    test_data['base_score'] = base_ensemble.get_alpha_score(test_data[selected_features])
    X_meta = test_data[selected_features].copy()
    X_meta['alpha_score'] = test_data['base_score']
    X_meta['daily_rank'] = test_data.groupby(level=0)['base_score'].rank(ascending=False)
    test_data['success_score'] = success_mdl.predict_trust(X_meta)
    test_data['similarity_alpha'] = sim_engine.get_similarity_feature(test_data[selected_features])

    test_data['final_score'] = (0.5 * test_data['base_score']) + \
                               (0.3 * test_data['success_score']) + \
                               (0.2 * (test_data['similarity_alpha'] > 0).astype(float))

    summary, daily = calculate_top_k_metrics(test_data, 'final_score', target_ret, k=5)

    print("\n" + "="*40)
    print("PHASE 9 UNSEEN FORWARD TEST RESULTS")
    print("="*40)
    print(f"Avg Hit Rate@5: {summary['avg_hit_rate']:.2%}")
    print(f"Avg 5D Return:  {summary['avg_return']:.2%}")
    print(f"Win Rate (Days): {summary['percent_days_positive']:.2%}")
    print("="*40)

    # 4. Final Comparison Report
    results = pd.DataFrame([{"Test": "Unseen Forward", **summary}])
    results.to_csv(DATA_REPORTS_DIR / "PHASE9_UNSEEN_RESULTS.csv", index=False)

if __name__ == "__main__":
    main()
