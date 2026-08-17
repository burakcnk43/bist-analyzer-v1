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
from app.ml.group_failure import GroupOutcomePredictorV2
from app.ml.market_quality import HighQualityMarketDayPredictor
from app.ml.adaptive_k import AdaptiveTopKSelectorV2
from app.ml.selection import Top5CombinationOptimizer
from app.ml.production_scorer import ProductionScorer
from app.ml.false_positive import FalsePositivePredictor
from app.ml.prob_tables import ConditionalProbEngine
from app.ml.deceptive_confidence import DeceptiveConfidenceDetector
from app.ml.monitoring import DriftGuard
from app.ml.interaction_miner import FeatureInteractionEngine
from app.ml.meta_labeler import AlphaTrustModel
from app.research.regimes import MarketRegimeDetector
from app.validation.evaluator import ResearchEvaluator
from app.validation.cpcv import CPCVAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase20Research")

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

def main():
    logger.info("--- PHASE 20: ROBUST ALPHA IMPROVEMENT & INVARIANT SIGNAL LEARNING ---")

    # 1. Load Data
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:100]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            all_data.append(df)
    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)

    # Feature Engineering
    box_engine = DarvasBoxEngine()
    processed_dfs = []
    for sym, group in global_df.groupby('symbol_col'):
        processed_dfs.append(box_engine.add_box_features(group))
    global_df = pd.concat(processed_dfs).sort_index()
    global_df = global_df.replace([np.inf, -np.inf], 0).fillna(0)
    breadth_df = breadth_df.replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Mining Invariant Interactions
    miner = FeatureInteractionEngine()
    # Merge breadth to global permanently for interaction mining and application
    global_df = global_df.join(breadth_df, how='left')
    global_df = global_df.replace([np.inf, -np.inf], 0).fillna(0)

    stable_features = ['relative_volume', 'dist_sma_20', 'pct_above_sma50', 'rsi_14']
    miner.discover_interactions(global_df, 'target_return_5d', stable_features, 'volatility_regime')
    global_df = miner.apply_interactions(global_df)

    # 3. Setup Components
    evaluator = ResearchEvaluator()
    regime_detector = MarketRegimeDetector()
    deception_detector = DeceptiveConfidenceDetector()
    p12_scorer = ProductionScorer(MODELS_DIR)
    combo_opt = Top5CombinationOptimizer()

    # Training Periods
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01'
    train_df = global_df[global_df.index <= train_end].copy()
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)].copy()
    test_df = global_df[global_df.index > meta_train_end].copy()

    # Train Specialists
    prob_engine = ConditionalProbEngine()
    prob_engine.fit(train_df, (train_df['target_return_5d'] > 0))

    fp_predictor = FalsePositivePredictor()
    meta_scored = []
    for date in meta_df.index.unique():
        day_data = meta_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})
        meta_scored.append(day_data.join(scores))
    full_meta = pd.concat(meta_scored)
    is_top = full_meta['production_alpha'] > full_meta.groupby(level=0)['production_alpha'].transform(lambda x: x.quantile(0.8))
    is_fail = full_meta['target_return_5d'] <= 0
    fp_predictor.train(clean_for_xgb(full_meta.drop(columns=['target_return_5d'], errors='ignore')), is_top, is_fail)

    # Strategy: Phase 20 (Full Stack)
    def phase20_selection(date, day_data):
        day_mkt = breadth_df.loc[[date]]
        regime = regime_detector.detect_regime(day_mkt)
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': regime})
        day_scored = day_data.join(scores)

        # P17-P18 Refinements
        day_scored = deception_detector.filter_candidates(day_scored, day_mkt.iloc[0])
        day_scored = prob_engine.apply_overrides(day_scored)

        return combo_opt.select_optimal_set(
            day_scored, k=5,
            failure_predictor=fp_predictor,
            regime=regime
        )

    logger.info("Executing Robustness Check...")
    m_p20, res_p20 = evaluator.evaluate_strategy(test_df, phase20_selection, global_df, regime_detector, breadth_df)

    print("\n--- PHASE 20 ROBUSTNESS SCOREBOARD ---")
    print(f"Base CPCV Median (P19): 51.12%")
    print(f"P20 Uncond 3PLUS (Recent): {m_p20['unconditional_3plus']:.2%}")
    print(f"Avg 5D Return: {m_p20['avg_5d_return']:.2%}")

    # Save results
    res_p20.to_csv(DATA_REPORTS_DIR / "PHASE20_DAILY_RESULTS.csv")

if __name__ == "__main__":
    main()
