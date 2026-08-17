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
logger = logging.getLogger("Phase9Optimization")

def main():
    logger.info("--- PHASE 9: OPTIMIZED META-ALPHA ---")

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

    max_date = global_df.index.max()
    unseen_start = max_date - timedelta(days=60)
    train_end = unseen_start - timedelta(days=20)

    train_df = global_df[global_df.index < train_end]
    test_df = global_df[global_df.index >= unseen_start]

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col', 'Unnamed'])])))

    # 2. Train Base Ensemble
    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    p_cls.fit(train_df[selected_features + [target_cls]].dropna()[selected_features],
             train_df[selected_features + [target_cls]].dropna()[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    p_rnk.fit(train_df[selected_features + [target_rnk]].dropna()[selected_features],
             train_df[selected_features + [target_rnk]].dropna()[target_rnk])

    base_ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

    # 3. Train Meta Models
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    log_df = pd.read_csv(log_path)
    log_df['date'] = pd.to_datetime(log_df['date'])
    log_df = log_df[log_df['date'] < train_end]

    # Success Predictor
    success_mdl = SuccessPredictor()
    success_mdl.train(log_df, selected_features)

    # Failure Predictor (Daily Level)
    # We aggregate the log by day
    daily_log = log_df.groupby('date').agg({
        'is_hit': 'mean',
        'macro_usdtry_return': 'first',
        'macro_bist100_return': 'first',
        'market_z_rsi_14': 'first'
    }).rename(columns={'is_hit': 'hit_rate'})

    fail_mdl = FailurePredictor()
    fail_mdl.train(daily_log)

    # 4. Final Evaluation with Dynamic Failure Discount
    logger.info("Evaluating Optimized Ensemble...")
    test_data = test_df[selected_features + [target_ret]].dropna().copy()

    test_data['base'] = base_ensemble.get_alpha_score(test_data[selected_features])

    X_meta = test_data[selected_features].copy()
    X_meta['alpha_score'] = test_data['base']
    X_meta['daily_rank'] = test_data.groupby(level=0)['base'].rank(ascending=False)
    test_data['success'] = success_mdl.predict_trust(X_meta)

    # Daily Failure Probability
    daily_feats = test_data.groupby(level=0)[fail_mdl.features].first()
    daily_fail_probs = fail_mdl.predict_failure_prob(daily_feats)
    test_data['fail_prob'] = test_data.index.map(pd.Series(daily_fail_probs, index=daily_feats.index))

    # Optimized Score: Base + Success - Failure Risk
    test_data['opt_alpha'] = (0.5 * test_data['base']) + (0.4 * test_data['success']) - (0.2 * test_data['fail_prob'])

    # 5. Compare
    base_sum, _ = calculate_top_k_metrics(test_data, 'base', target_ret, k=5)
    opt_sum, _ = calculate_top_k_metrics(test_data, 'opt_alpha', target_ret, k=5)

    print("\n" + "="*40)
    print("PHASE 9 OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Base Hit Rate@5: {base_sum['avg_hit_rate']:.2%}")
    print(f"Opt Hit Rate@5:  {opt_sum['avg_hit_rate']:.2%}")
    print(f"Base Avg Return: {base_sum['avg_return']:.2%}")
    print(f"Opt Avg Return:  {opt_sum['avg_return']:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
