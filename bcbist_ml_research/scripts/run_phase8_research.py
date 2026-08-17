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
logger = logging.getLogger("Phase8Research")

def main():
    logger.info("--- PHASE 8: META-LEARNING & ALPHA OPTIMIZATION ---")

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

    # 2. Train Base Model (Phase 7 equivalent)
    logger.info("Training Base Models (Phase 7 Baseline)...")
    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    tr_c = research_df[selected_features + [target_cls]].dropna()
    p_cls.fit(tr_c[selected_features], tr_c[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    tr_r = research_df[selected_features + [target_rnk]].dropna()
    p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

    base_ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

    # 3. Success Meta-Model Training
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    if not log_path.exists():
        logger.error("Run scripts/phase8_generate_prediction_log.py first.")
        return

    log_df = pd.read_csv(log_path)
    success_mdl = SuccessPredictor()
    success_mdl.train(log_df, selected_features)

    # 4. Similarity Engine
    sim_engine = SimilarityEngine(k=20)
    sim_engine.fit(research_df.dropna(subset=[target_ret]), target_ret)

    # 5. Full Phase 8 Evaluation on Holdout
    logger.info("Running Phase 8 Meta-Alpha Evaluation...")

    test_data = holdout_df[selected_features + [target_ret]].dropna().copy()

    # Signal A: Base Score
    test_data['base_score'] = base_ensemble.get_alpha_score(test_data[selected_features])

    # Signal B: Success Pattern (Meta)
    X_meta = test_data[selected_features].copy()
    X_meta['alpha_score'] = test_data['base_score']
    X_meta['daily_rank'] = test_data.groupby(level=0)['base_score'].rank(ascending=False)
    test_data['success_score'] = success_mdl.predict_trust(X_meta)

    # Signal C: Similarity (Evidence)
    test_data['similarity_alpha'] = sim_engine.get_similarity_feature(test_data[selected_features])

    # Final Meta-Ensemble
    test_data['meta_alpha'] = (0.5 * test_data['base_score']) + \
                             (0.3 * test_data['success_score']) + \
                             (0.2 * (test_data['similarity_alpha'] > 0).astype(float))

    # 6. Metrics
    p7_summary, _ = calculate_top_k_metrics(test_data, 'base_score', target_ret, k=5)
    p8_summary, _ = calculate_top_k_metrics(test_data, 'meta_alpha', target_ret, k=5)

    print("\n" + "="*40)
    print("PHASE 8 RESEARCH SUMMARY")
    print("="*40)
    print(f"P7 Hit Rate@5: {p7_summary['avg_hit_rate']:.2%}")
    print(f"P8 Hit Rate@5: {p8_summary['avg_hit_rate']:.2%}")
    print(f"P7 Avg Return: {p7_summary['avg_return']:.2%}")
    print(f"P8 Avg Return: {p8_summary['avg_return']:.2%}")
    print(f"Improvement:   {p8_summary['avg_hit_rate'] - p7_summary['avg_hit_rate']:.2%}")
    print("="*40)

    # 7. Generate Full Report
    generate_final_report(p7_summary, p8_summary)

def generate_final_report(p7, p8):
    report = "# Phase 8: Success Pattern Learning & Optimization\n\n"
    report += "## 1. Meta-Alpha Performance\n"
    report += "| Metric | Phase 7 (Base) | Phase 8 (Meta) | Delta |\n"
    report += "|:---|:---:|:---:|:---:|\n"
    report += f"| Hit Rate@5 | {p7['avg_hit_rate']:.2%} | {p8['avg_hit_rate']:.2%} | {p8['avg_hit_rate'] - p7['avg_hit_rate']:+.2%} |\n"
    report += f"| Avg 5D Return | {p7['avg_return']:.2%} | {p8['avg_return']:.2%} | {p8['avg_return'] - p7['avg_return']:+.2%} |\n"
    report += f"| Days Positive | {p7['percent_days_positive']:.2%} | {p8['percent_days_positive']:.2%} | {p8['percent_days_positive'] - p7['percent_days_positive']:+.2%} |\n\n"

    report += "## 2. Meta-Learning components\n"
    report += "- **Success Model**: XGBoost trained on OOS validation hit patterns.\n"
    report += "- **Similarity Engine**: k-NN evidence from 5 years of historical outcomes.\n"

    with open(DATA_REPORTS_DIR / "PHASE8_FINAL_REPORT.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
