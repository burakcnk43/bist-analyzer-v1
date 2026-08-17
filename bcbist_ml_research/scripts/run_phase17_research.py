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
from app.ml.group_failure import GroupFailurePredictor
from app.ml.deceptive_confidence import DeceptiveConfidenceDetector
from app.ml.prob_tables import ConditionalProbEngine
from app.ml.false_positive import FalsePositivePredictor
from app.ml.box_conditional import DarvasConditionalExpert
from app.ml.sector_rules import SectorConditionalSuccessModel
from app.ml.selection import Top5CombinationOptimizer
from app.ml.production_scorer import ProductionScorer
from app.research.regimes import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase17Research")

def main():
    logger.info("--- PHASE 17: DEEP FAILURE DISCOVERY & CONDITIONAL PROBABILITY OPTIMIZATION ---")

    # 1. Load Data
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:150]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            all_data.append(df)
    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)

    # 2. Complete Phase 16 Autopsy Feature Engineering
    logger.info("Adding Phase 17 Specific Features...")
    box_engine = DarvasBoxEngine()
    # Batch box features
    processed_dfs = []
    for sym, group in global_df.groupby('symbol_col'):
        processed_dfs.append(box_engine.add_box_features(group))
    global_df = pd.concat(processed_dfs).sort_index()
    global_df = global_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Splits
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01'
    test_end = '2026-08-12'

    train_df = global_df[global_df.index <= train_end].copy()
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)].copy()
    test_df = global_df[global_df.index > meta_train_end].copy()

    # 3. Train Phase 17 Specialists
    logger.info("Training Intelligence Layer...")

    # Conditional Prob Tables
    prob_engine = ConditionalProbEngine()
    prob_engine.fit(train_df, (train_df['target_return_5d'] > 0))

    # Group Failure Predictor (Need historical group outcomes)
    p12_scorer = ProductionScorer(MODELS_DIR)
    combo_opt = Top5CombinationOptimizer()

    group_records = []
    meta_dates = sorted(meta_df.index.unique())
    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        day_mkt = breadth_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})

        # Select with P16 logic (Simplified)
        top5 = combo_opt.select_optimal_set(day_data.join(scores), k=5)

        # Outcome
        hits = (meta_df.loc[[date]].set_index('symbol_col').loc[top5['symbol_col'].tolist()]['target_return_5d'] > 0).sum()

        # Feature for failure model
        gf_features = GroupFailurePredictor().prepare_group_features(day_mkt.iloc[0], day_data.join(scores))
        gf_features['failed_3plus'] = int(hits < 3)
        group_records.append(gf_features)

    group_history = pd.concat(group_records)
    group_failure_model = GroupFailurePredictor()
    group_failure_model.train(group_history.drop(columns=['failed_3plus']), group_history['failed_3plus'])

    # False Positive Lab (P16)
    full_meta = []
    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})
        full_meta.append(day_data.join(scores))
    full_meta_df = pd.concat(full_meta)

    fp_predictor = FalsePositivePredictor()
    # Mock clean
    X_fp = full_meta_df.select_dtypes(include=[np.number]).drop(columns=['target_return_5d'], errors='ignore')
    is_top = full_meta_df['production_alpha'] > full_meta_df.groupby(level=0)['production_alpha'].transform(lambda x: x.quantile(0.8))
    is_fail = full_meta_df['target_return_5d'] <= 0
    fp_predictor.train(X_fp, is_top, is_fail)

    # 4. Final Evaluation with Full Pipeline
    logger.info("Evaluating Phase 17 vs Phase 16...")
    test_dates = sorted(test_df.index.unique())
    results = []

    regime_detector = MarketRegimeDetector()
    deception_detector = DeceptiveConfidenceDetector()

    for date in test_dates:
        day_df_raw = test_df.loc[[date]].copy()
        day_mkt = breadth_df.loc[[date]]
        regime = regime_detector.detect_regime(day_mkt)

        # P15 Baseline scores
        scores = p12_scorer.calculate_production_scores(day_df_raw, {'regime': regime})
        day_scored = day_df_raw.join(scores)

        # P17 Enhancements:
        # 1. Deceptive Confidence Filter
        day_scored = deception_detector.filter_candidates(day_scored, day_mkt.iloc[0])

        # 2. Conditional Table Overrides
        day_scored = prob_engine.apply_overrides(day_scored)

        # 3. Group Failure Check
        group_X = group_failure_model.prepare_group_features(day_mkt.iloc[0], day_scored)
        p_group_fail = group_failure_model.predict_failure_prob(group_X)

        # Selection
        top5_p17 = combo_opt.select_optimal_set(
            day_scored, k=5,
            failure_predictor=fp_predictor,
            regime=regime
        )

        # Outcomes
        actuals = global_df.loc[[date]].set_index('symbol_col')
        h17 = (actuals.loc[top5_p17['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_p17.empty else 0

        # Compare with P16 results from file if available or recalculate
        # For simplicity, we re-run P16 on the same data
        top5_p16 = combo_opt.select_optimal_set(day_scored, k=5)
        h16 = (actuals.loc[top5_p16['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_p16.empty else 0

        results.append({
            'date': date,
            'p17_hits': h17,
            'p16_hits': h16,
            'p_group_fail': p_group_fail,
            'regime': regime
        })

    res_df = pd.DataFrame(results)

    print(f"\n--- PHASE 17 SCOREBOARD ---")
    print(f"PHASE 16 Top5_3PLUS = {(res_df['p16_hits'] >= 3).mean():.2%}")
    print(f"PHASE 17 Top5_3PLUS = {(res_df['p17_hits'] >= 3).mean():.2%}")
    print(f"DELTA (3PLUS) = {(res_df['p17_hits'] >= 3).mean() - (res_df['p16_hits'] >= 3).mean():+.2%}")
    print(f"Avg P(Group Failure) = {res_df['p_group_fail'].mean():.2%}")

    res_df.to_csv(DATA_REPORTS_DIR / "PHASE17_DAILY_RESULTS.csv")

if __name__ == "__main__":
    main()
