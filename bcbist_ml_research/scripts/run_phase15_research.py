import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, DATA_REPORTS_DIR, MODELS_DIR
from app.features.boxes import DarvasBoxEngine
from app.ml.market_context import MarketContextManager
from app.ml.hit_rate import DailyHitRatePredictor, DailyConfidenceGate
from app.ml.clusters import SuccessDayClusterEngine, FailureDayClusterEngine
from app.ml.conditional import ConditionalStockScorer
from app.ml.selection import Top5CombinationOptimizer
from app.ml.production_scorer import ProductionScorer
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase15Research")

def align_features(df, features):
    df_copy = df.copy()
    actual_cols = {c.lower().strip(): c for c in df_copy.columns}
    final_df = pd.DataFrame(index=df_copy.index)
    for f in features:
        f_key = f.lower().strip()
        if f_key in actual_cols:
            final_df[f] = df_copy[actual_cols[f_key]]
        else:
            final_df[f] = 0.0
    return final_df

def main():
    logger.info("--- PHASE 15: ADAPTIVE TOP5_3PLUS HIT-RATE OPTIMIZATION ---")

    # 1. Data Preparation
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:150]: # Top 200 for speed in research
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            all_data.append(df)

    global_df = pd.concat(all_data).sort_index()
    logger.info(f"Loaded {len(global_df)} rows.")

    # Load market features
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)
    sector_df = pd.read_csv('data/sector_rotation.csv', parse_dates=['date'])

    # Define splits
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01' # Period for training HitRate model
    test_end = '2026-08-05'

    train_df = global_df[global_df.index <= train_end]
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)]
    test_df = global_df[global_df.index > meta_train_end]

    # 2. Phase 13/14 Components Preparation
    # (Simplified: we assume we have the base ensemble from models/ directory)
    p12_scorer = ProductionScorer(MODELS_DIR)
    base_ens = p12_scorer.models.get(5)

    # 3. Generate Daily Hit Labels for Meta-Training
    logger.info("Generating historical hit log for DailyHitRatePredictor...")
    hit_records = []
    meta_dates = sorted(meta_df.index.unique())

    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        # Use simple baseline selection to see which days 'work' naturally
        # In real life, we'd use the full P13 stack here
        top5 = day_data.sort_values('market_z_rsi_14', ascending=False).head(5) # Mock selection
        hits = (top5['target_return_5d'] > 0).sum()
        hit_records.append({'date': date, 'hits': hits, 'is_3plus': int(hits >= 3)})

    hit_df = pd.DataFrame(hit_records).set_index('date')

    # 4. Train Phase 15 Models
    logger.info("Training Phase 15 Intelligence Layer...")

    # Market Features for HitRate model
    mkt_feats = breadth_df.reindex(meta_dates).fillna(0).replace([np.inf, -np.inf], 0)

    hit_predictor = DailyHitRatePredictor()
    hit_predictor.train(mkt_feats, hit_df['is_3plus'])

    success_clusters = SuccessDayClusterEngine()
    success_clusters.fit(mkt_feats, hit_df['hits'])

    fail_clusters = FailureDayClusterEngine()
    fail_clusters.fit(mkt_feats, hit_df['hits'])

    cond_scorer = ConditionalStockScorer()
    # Training conditional scorer requires stock + market features
    # Combining meta_df with breadth for sample dates
    combined_meta = meta_df.join(breadth_df, how='inner').replace([np.inf, -np.inf], 0)

    # Drop all non-numeric columns and targets
    X_cond = combined_meta.select_dtypes(include=[np.number]).drop(columns=[c for c in combined_meta.columns if 'target' in c], errors='ignore')

    cond_scorer.train(
        X_cond,
        (combined_meta['target_return_5d'] > 0),
        combined_meta.index.map(hit_df['is_3plus']).fillna(0)
    )

    # Pre-clean test_df and breadth_df
    test_df = test_df.replace([np.inf, -np.inf], 0)
    breadth_df = breadth_df.replace([np.inf, -np.inf], 0)

    # 5. Final Evaluation (OOS)
    logger.info("Evaluating Phase 15 vs Phase 13...")
    test_dates = sorted(test_df.index.unique())
    results = []

    combo_opt = Top5CombinationOptimizer()
    conf_gate = DailyConfidenceGate()

    for date in test_dates:
        day_df_raw = test_df.loc[[date]].copy()
        day_mkt = breadth_df.loc[[date]]

        # Predicted Hit Rate
        p_3plus = hit_predictor.predict_hit_prob(day_mkt)
        mode = conf_gate.get_selection_mode(p_3plus)

        # Calculate production_alpha using P13 stack
        mkt_ctx = {'regime': 'NORMAL', 'transition_prob': 0.0} # Simplified for research
        scored_df = p12_scorer.calculate_production_scores(day_df_raw, mkt_ctx)

        # Group Selection
        top5_p15 = combo_opt.select_optimal_set(
            scored_df.join(day_df_raw[['symbol_col', 'sector']], how='inner'),
            k=5,
            conditional_scorer=cond_scorer,
            market_vector=day_mkt
        )

        # Hits calculation
        actuals = global_df.loc[[date]].set_index('symbol_col')
        h15 = (actuals.loc[top5_p15['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_p15.empty else 0

        results.append({
            'date': date,
            'p_3plus': p_3plus,
            'mode': mode,
            'hits': h15
        })

    res_df = pd.DataFrame(results)
    p15_3plus = (res_df['hits'] >= 3).mean()

    print(f"\n--- PHASE 15 RESULTS ---")
    print(f"Top5_3PLUS Hit Rate: {p15_3plus:.2%}")
    print(f"Avg Predicted P(3PLUS): {res_df['p_3plus'].mean():.2%}")
    print(f"Modes: {res_df['mode'].value_counts().to_dict()}")

    # Save final comparison
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE15_DAILY_RESULTS.csv")

if __name__ == "__main__":
    main()
