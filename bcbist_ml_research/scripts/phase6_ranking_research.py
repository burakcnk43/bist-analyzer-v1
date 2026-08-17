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
from app.validation.walk_forward import WalkForwardValidation
from app.ml.pipeline import ResearchPipeline
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RankingResearch")

def main():
    logger.info("--- PHASE 6: DAILY TOP-5 RANKING RESEARCH ---")

    # 1. Load All Features
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    # Holdout setup
    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    # Task configuration
    h = 5 # Start with 5-day horizon for ranking
    target_rank_label = f'target_label_{h}d' # Integer classes for LTR
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # --- EXPERIMENT 1: CLASSIFICATION VS RANKING ---
    tasks = [
        ("Binary Classifier", "classification", f'target_up_{h}d'),
        ("XGBRanker (LTR)", "ranking", target_rank_label)
    ]

    final_results = []

    for task_name, task_type, target_col in tasks:
        logger.info(f"Running Task: {task_name}")

        # Train on research data
        pipeline = ResearchPipeline(model_type='xgboost', n_features=30, task=task_type)
        train_clean = research_df[selected_features + [target_col]].dropna()
        pipeline.fit(train_clean[selected_features], train_clean[target_col])

        # Evaluate on Holdout
        test_data = holdout_df[selected_features + [target_ret]].copy()
        test_data = test_data.dropna()

        # Get scores
        scores = pipeline.predict_proba(test_data[selected_features])
        test_data['model_score'] = scores

        # Metrics for Top 5
        summary, daily = calculate_top_k_metrics(test_data, 'model_score', target_ret, k=5)

        logger.info(f"  {task_name} Top-5 Result:")
        logger.info(f"    Avg Return: {summary['avg_return']:.2%}")
        logger.info(f"    Excess Return: {summary['avg_excess_return']:.2%}")
        logger.info(f"    Hit Rate@5: {summary['avg_hit_rate']:.2%}")

        final_results.append({
            "Task": task_name,
            "Avg_Return": summary['avg_return'],
            "Excess_Ret": summary['avg_excess_return'],
            "Hit_Rate_At_5": summary['avg_hit_rate']
        })

    res_df = pd.DataFrame(final_results)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE6_RANKING_RESULTS.csv", index=False)

    print("\nPhase 6 Ranking Comparison:")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
