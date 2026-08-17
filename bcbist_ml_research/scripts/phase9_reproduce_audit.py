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
from app.ml.ensemble import AlphaEnsemble
from app.ml.meta_models import SuccessPredictor
from app.features.similarity import SimilarityEngine
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase9Reproduction")

def main():
    logger.info("--- PHASE 9: REPRODUCIBILITY & ROBUSTNESS AUDIT ---")

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

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col', 'symbol', 'sector']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # Load Prediction Log for Meta training
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    if not log_path.exists():
        logger.error("Run scripts/phase8_generate_prediction_log.py first.")
        return
    log_df = pd.read_csv(log_path)

    # 2. Reproduction of Phase 8
    logger.info("Training Frozen Phase 8 components...")

    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    tr_c = research_df[selected_features + [target_cls]].dropna()
    p_cls.fit(tr_c[selected_features], tr_c[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    tr_r = research_df[selected_features + [target_rnk]].dropna()
    p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

    base_ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

    success_mdl = SuccessPredictor()
    success_mdl.train(log_df, selected_features)

    sim_engine = SimilarityEngine(k=20)
    sim_engine.fit(research_df.dropna(subset=[target_ret]), target_ret)

    # 3. Evaluate Holdout
    logger.info("Evaluating Phase 8 on Holdout (Reproduction)...")
    test_data = holdout_df[selected_features + [target_ret]].dropna().copy()

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

    logger.info(f"REPRODUCED Hit Rate@5: {summary['avg_hit_rate']:.2%}")
    logger.info(f"REPRODUCED Avg Return: {summary['avg_return']:.2%}")

    # 4. Save Prediction Audit
    test_data.to_csv(DATA_REPORTS_DIR / "PHASE9_PREDICTION_AUDIT.csv")

    # 5. Seed Robustness Test (10 seeds)
    logger.info("Running Seed Robustness Test (10 seeds)...")
    seed_results = []
    for s in range(10):
        # We vary the seed for the Base Ranker as it's the primary signal
        p_rnk_s = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
        p_rnk_s.model.params['random_state'] = s
        p_rnk_s.fit(tr_r[selected_features], tr_r[target_rnk])

        ens_s = AlphaEnsemble(p_cls, p_rnk_s, w_cls=0.3, w_rnk=0.7)
        scores_s = ens_s.get_alpha_score(test_data[selected_features])

        # Calculate hit rate with this seed
        test_data_s = test_data.copy()
        test_data_s['alpha_s'] = scores_s
        sum_s, _ = calculate_top_k_metrics(test_data_s, 'alpha_s', target_ret, k=5)
        seed_results.append(sum_s['avg_hit_rate'])

    logger.info(f"Seed Robustness: Mean={np.mean(seed_results):.2%}, Std={np.std(seed_results):.2%}")

    pd.DataFrame({"seed_hit_rate": seed_results}).to_csv(DATA_REPORTS_DIR / "PHASE9_SEED_ROBUSTNESS.csv", index=False)

if __name__ == "__main__":
    main()
