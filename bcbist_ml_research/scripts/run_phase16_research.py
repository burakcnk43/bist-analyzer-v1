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
from app.ml.false_positive import FalsePositivePredictor
from app.ml.box_conditional import DarvasConditionalExpert
from app.ml.sector_rules import SectorConditionalSuccessModel
from app.ml.selection import Top5CombinationOptimizer
from app.ml.production_scorer import ProductionScorer
from app.ml.conditional import ConditionalStockScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase16Research")

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
    logger.info("--- PHASE 16: TOP-5 GROUP HIT-RATE & CONDITIONAL SPECIALIST OPTIMIZATION ---")

    # 1. Load Data
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:150]: # Reduced for speed
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            all_data.append(df)
    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)

    # 2. Add Missing Box Features
    logger.info("Calculating Structural Box Features...")
    box_engine = DarvasBoxEngine()
    processed_dfs = []
    for sym, group in global_df.groupby('symbol_col'):
        processed_dfs.append(box_engine.add_box_features(group))
    global_df = pd.concat(processed_dfs).sort_index()
    global_df = global_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Splits
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01'
    test_end = '2026-08-05'

    train_df = global_df[global_df.index <= train_end].copy()
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)].copy()
    test_df = global_df[global_df.index > meta_train_end].copy()

    # 3. Train Phase 16 Specialists
    logger.info("Training Phase 16 Specialized Models...")

    def clean_for_xgb(df):
        df_clean = df.copy()
        regime_map = {
            "BULL": 1, "BULL_OVEREXTENDED": 2, "BULL_TO_SIDEWAYS": 0.5,
            "SIDEWAYS_HIGH_VOL": 0, "SIDEWAYS_LOW_VOL": 0,
            "SIDEWAYS_TO_BEAR": -0.5, "BEAR": -1, "BEAR_TO_CRASH": -1.5,
            "CRASH": -2, "CRASH_TO_RECOVERY": -1, "RECOVERY": 0.5,
            "NORMAL": 0
        }
        if 'regime' in df_clean.columns:
            df_clean['regime_val'] = df_clean['regime'].map(regime_map).fillna(0)

        drop_cols = ['regime', 'symbol_col', 'sector', 'date', 'symbol']
        df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
        return df_clean.select_dtypes(include=[np.number])

    # False Positive Lab
    p12_scorer = ProductionScorer(MODELS_DIR)

    meta_results = []
    for date in meta_df.index.unique():
        day_data = meta_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL', 'transition_prob': 0.0})
        day_data = day_data.join(scores)
        meta_results.append(day_data)

    full_meta = pd.concat(meta_results)
    is_top = full_meta['production_alpha'] > full_meta.groupby(level=0)['production_alpha'].transform(lambda x: x.quantile(0.8))
    is_fail = full_meta['target_return_5d'] <= 0

    fp_predictor = FalsePositivePredictor()
    fp_predictor.train(clean_for_xgb(full_meta.drop(columns=['target_return_5d'], errors='ignore')), is_top, is_fail)

    # Conditional Box Specialist
    box_specialist = DarvasConditionalExpert()
    box_specialist.train(train_df, (train_df['target_return_5d'] > 0))

    # Sector Specialist
    sector_specialist = SectorConditionalSuccessModel()
    sector_specialist.train_sector_models(
        train_df.drop(columns=['target_return_5d'], errors='ignore'),
        (train_df['target_return_5d'] > 0)
    )

    # 4. Final Evaluation with Upgraded Optimizer
    logger.info("Evaluating Phase 16 vs Phase 15...")
    test_dates = sorted(test_df.index.unique())
    results = []

    combo_opt = Top5CombinationOptimizer()

    for date in test_dates:
        day_df_raw = test_df.loc[[date]].copy()

        # P15 Baseline scores
        mkt_ctx = {'regime': 'NORMAL', 'transition_prob': 0.0}
        scored_df = p12_scorer.calculate_production_scores(day_df_raw, mkt_ctx)
        day_info = day_df_raw[['symbol_col', 'sector']]
        scored_full = scored_df.join(day_info, how='inner')

        # Selection A: Phase 15 (Greedy Sector)
        top5_p15 = combo_opt.select_optimal_set(scored_full, k=5)

        # Phase 16 Scoring Logic
        p_sector = sector_specialist.predict_probs(day_df_raw)
        p_box = box_specialist.predict_success_prob(day_df_raw)

        day_df_raw['p_sector'] = p_sector
        day_df_raw['p_box'] = p_box

        scored_p16 = scored_full.join(day_df_raw[['p_sector', 'p_box']], how='inner')
        scored_p16['production_alpha'] = (
            0.4 * scored_p16['production_alpha'] +
            0.3 * scored_p16['p_sector'] +
            0.3 * scored_p16['p_box']
        )

        # Selection B: Phase 16 (Correlation Aware + False Positive + Conditional)
        top5_p16 = combo_opt.select_optimal_set(
            scored_p16, k=5,
            failure_predictor=fp_predictor
        )

        # Outcomes
        actuals = global_df.loc[[date]].set_index('symbol_col')
        h15 = (actuals.loc[top5_p15['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_p15.empty else 0
        h16 = (actuals.loc[top5_p16['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_p16.empty else 0

        results.append({
            'date': date,
            'p15_hits': h15,
            'p16_hits': h16,
            'p16_ret': actuals.loc[top5_p16['symbol_col'].tolist()]['target_return_5d'].mean() if not top5_p16.empty else 0
        })

    res_df = pd.DataFrame(results)

    print(f"\n--- PHASE 16 SCOREBOARD ---")
    print(f"PHASE 15 Top5_3PLUS = {(res_df['p15_hits'] >= 3).mean():.2%}")
    print(f"PHASE 16 Top5_3PLUS = {(res_df['p16_hits'] >= 3).mean():.2%}")
    print(f"PHASE 16 Top5_4PLUS = {(res_df['p16_hits'] >= 4).mean():.2%}")
    print(f"PHASE 16 Avg Return  = {res_df['p16_ret'].mean():.2%}")
    print(f"DELTA (3PLUS) = {(res_df['p16_hits'] >= 3).mean() - (res_df['p15_hits'] >= 3).mean():+.2%}")

    res_df.to_csv(DATA_REPORTS_DIR / "PHASE16_DAILY_RESULTS.csv")

if __name__ == "__main__":
    main()
