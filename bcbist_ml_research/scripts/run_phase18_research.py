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
from app.ml.adaptive_k import AdaptiveTopKSelector
from app.ml.selection import Top5CombinationOptimizer
from app.ml.production_scorer import ProductionScorer
from app.ml.false_positive import FalsePositivePredictor
from app.ml.prob_tables import ConditionalProbEngine
from app.ml.deceptive_confidence import DeceptiveConfidenceDetector
from app.research.regimes import MarketRegimeDetector
from app.validation.evaluator import ResearchEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase18Research")

def main():
    logger.info("--- PHASE 18: ADAPTIVE TOP-K SELECTION & PHASE 17 VALIDATION ---")

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

    # 2. Add Box Features
    box_engine = DarvasBoxEngine()
    processed_dfs = []
    for sym, group in global_df.groupby('symbol_col'):
        processed_dfs.append(box_engine.add_box_features(group))
    global_df = pd.concat(processed_dfs).sort_index()
    global_df = global_df.replace([np.inf, -np.inf], 0).fillna(0)
    breadth_df = breadth_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Splits
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01'
    test_end = '2026-08-12'

    train_df = global_df[global_df.index <= train_end].copy()
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)].copy()
    test_df = global_df[global_df.index > meta_train_end].copy()

    # 3. Train Models (PIT Compliance)
    logger.info("Training Group Intelligence stack...")

    prob_engine = ConditionalProbEngine()
    prob_engine.fit(train_df, (train_df['target_return_5d'] > 0))

    p12_scorer = ProductionScorer(MODELS_DIR)
    combo_opt = Top5CombinationOptimizer()

    # Generate Training data for GroupOutcomePredictorV2 using meta_df
    group_training_data = []
    meta_dates = sorted(meta_df.index.unique())
    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        day_mkt = breadth_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})

        day_full = day_data.join(scores)
        top5 = combo_opt.select_optimal_set(day_full, k=5)

        # Outcome
        hits = (meta_df.loc[[date]].set_index('symbol_col').loc[top5['symbol_col'].tolist()]['target_return_5d'] > 0).sum()

        # Features
        predictor = GroupOutcomePredictorV2()
        feats = predictor.prepare_group_features(day_mkt.iloc[0], day_full)
        feats['hit_count'] = hits
        group_training_data.append(feats)

    group_history = pd.concat(group_training_data)
    group_outcome_model = GroupOutcomePredictorV2()
    group_outcome_model.train(group_history.drop(columns=['hit_count']), group_history['hit_count'])

    # Train False Positive Predictor (P17 feature)
    full_meta = []
    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        scores = p12_scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})
        full_meta.append(day_data.join(scores))
    full_meta_df = pd.concat(full_meta)

    fp_predictor = FalsePositivePredictor()
    X_fp = full_meta_df.select_dtypes(include=[np.number]).drop(columns=['target_return_5d'], errors='ignore')
    is_top = full_meta_df['production_alpha'] > full_meta_df.groupby(level=0)['production_alpha'].transform(lambda x: x.quantile(0.8))
    is_fail = full_meta_df['target_return_5d'] <= 0
    fp_predictor.train(X_fp, is_top, is_fail)

    # 4. Evaluation Loop
    evaluator = ResearchEvaluator()
    regime_detector = MarketRegimeDetector()
    deception_detector = DeceptiveConfidenceDetector()
    quality_predictor = HighQualityMarketDayPredictor()
    k_selector = AdaptiveTopKSelector()

    # Base model used for selection
    def get_scored_candidates(date, day_df_raw):
        day_mkt = breadth_df.loc[[date]]
        regime = regime_detector.detect_regime(day_mkt)
        scores = p12_scorer.calculate_production_scores(day_df_raw, {'regime': regime})
        day_scored = day_df_raw.join(scores)

        # Apply P17-P18 Overrides
        day_scored = deception_detector.filter_candidates(day_scored, day_mkt.iloc[0])
        day_scored = prob_engine.apply_overrides(day_scored)
        return day_scored, day_mkt.iloc[0], regime

    # FORCED TOP-5 STRATEGY
    def forced_top5(date, day_data):
        scored, mkt_row, regime = get_scored_candidates(date, day_data)
        return combo_opt.select_optimal_set(
            scored, k=5,
            failure_predictor=fp_predictor,
            regime=regime
        )

    # ADAPTIVE TOP-K STRATEGY
    def adaptive_topk(date, day_data):
        scored, mkt_row, regime = get_scored_candidates(date, day_data)

        # Estimate quality and hit distribution
        quality_score = quality_predictor.calculate_quality_score(mkt_row, regime)
        group_X = group_outcome_model.prepare_group_features(mkt_row, scored)
        hit_dist = group_outcome_model.predict_hit_distribution(group_X)

        # Determine K
        k = k_selector.determine_optimal_k(hit_dist, quality_score)

        if k == 0: return pd.DataFrame()

        # Get ranked set
        top_k = combo_opt.select_optimal_set(
            scored, k=k,
            failure_predictor=fp_predictor,
            regime=regime
        )
        return top_k

    # Run Comparisons
    logger.info("Executing Strategy Comparisons...")
    m1, res1 = evaluator.evaluate_strategy(test_df, forced_top5, global_df)
    m2, res2 = evaluator.evaluate_strategy(test_df, adaptive_topk, global_df)

    print("\n--- PHASE 18 RESEARCH SCOREBOARD ---")
    print(f"Strategy: FORCED TOP-5")
    print(f"3PLUS Hit Rate (Unconditional): {m1['unconditional_3plus']:.2%}")
    print(f"Coverage: {m1['coverage']:.2%}")
    print(f"Avg 5D Return: {m1['avg_5d_return']:.2%}")

    print(f"\nStrategy: ADAPTIVE TOP-K")
    print(f"3PLUS Hit Rate (Conditional):   {m2['conditional_3plus']:.2%}")
    print(f"3PLUS Hit Rate (Unconditional): {m2['unconditional_3plus']:.2%}")
    print(f"Coverage: {m2['coverage']:.2%}")
    print(f"Average K: {m2['avg_k']:.2f}")
    print(f"Avg 5D Return: {m2['avg_5d_return']:.2%}")

    # Save results
    res1.to_csv(DATA_REPORTS_DIR / "PHASE18_FORCED_RESULTS.csv")
    res2.to_csv(DATA_REPORTS_DIR / "PHASE18_ADAPTIVE_RESULTS.csv")

if __name__ == "__main__":
    main()
