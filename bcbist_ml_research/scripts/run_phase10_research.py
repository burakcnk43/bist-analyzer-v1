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
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble
from app.ml.meta_models import SuccessPredictor
from app.features.similarity import SimilarityEngine
from app.ml.production_scorer import ProductionScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase10Research")

def main():
    logger.info("--- PHASE 10: PRODUCTION INTELLIGENCE RESEARCH ---")

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

    max_date = global_df.index.max()
    unseen_start = max_date - timedelta(days=60)
    train_end = unseen_start - timedelta(days=20)

    research_df = global_df[global_df.index < train_end]
    unseen_df = global_df[global_df.index >= unseen_start]

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    # 2. PHASE 9 BASELINE (The one that works)
    logger.info("Setting up high-performance Baseline...")
    p_cls_9 = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    p_cls_9.fit(research_df.dropna(subset=['target_up_5d'])[selected_features], research_df.dropna(subset=['target_up_5d'])['target_up_5d'])
    p_rnk_9 = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    p_rnk_9.fit(research_df.dropna(subset=['target_label_5d'])[selected_features], research_df.dropna(subset=['target_label_5d'])['target_label_5d'])
    ens_9 = AlphaEnsemble(p_cls_9, p_rnk_9, w_cls=0.3, w_rnk=0.7)

    # SAVE THIS TO MODELS_DIR for Production Scorer to use
    joblib.dump(p_cls_9, MODELS_DIR / "phase10_cls_5d.joblib")
    joblib.dump(p_rnk_9, MODELS_DIR / "phase10_rnk_5d.joblib")

    # 3. PHASE 10 PRODUCTION SCORER
    logger.info("Setting up Phase 10 Production Scorer...")
    p10_scorer = ProductionScorer(MODELS_DIR)

    # 4. SIDE-BY-SIDE EVALUATION
    logger.info("Running Side-by-Side Comparison on UNSEEN period...")
    extra_cols = ['target_return_5d', 'symbol_col', 'sector_col', 'rolling_std_20', 'market_z_rsi_14']
    all_cols = list(set(selected_features + extra_cols))
    test_data = unseen_df[all_cols].dropna().copy()

    dates = test_data.index.unique()
    comparison_results = []

    for date in dates:
        day_df = test_data.loc[[date]]

        # --- PHASE 9 (Base only for now to ensure model parity) ---
        base_scores_9 = ens_9.get_alpha_score(day_df[selected_features])
        day_df_9 = day_df.copy()
        day_df_9['final_score'] = base_scores_9
        top5_9 = day_df_9.sort_values('final_score', ascending=False).head(5)

        # --- PHASE 10 ---
        mkt_ctx = {'market_z_rsi_14': day_df['market_z_rsi_14'].iloc[0]}
        scores_10 = p10_scorer.calculate_production_scores(day_df[selected_features], mkt_ctx)
        top5_10 = p10_scorer.select_top_5(scores_10, day_df[['symbol_col', 'sector_col']])
        top5_10 = top5_10.merge(day_df[['symbol_col', 'target_return_5d']], on='symbol_col', how='left')

        comparison_results.append({
            "date": date,
            "p9_hit_rate": (top5_9['target_return_5d'] > 0).mean(),
            "p10_hit_rate": (top5_10['target_return_5d'] > 0).mean(),
            "p9_return": top5_9['target_return_5d'].mean(),
            "p10_return": top5_10['target_return_5d'].mean()
        })

    res_df = pd.DataFrame(comparison_results)

    print("\n" + "="*40)
    print(f"P9 Avg Hit Rate: {res_df['p9_hit_rate'].mean():.2%}")
    print(f"P10 Avg Hit Rate: {res_df['p10_hit_rate'].mean():.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
