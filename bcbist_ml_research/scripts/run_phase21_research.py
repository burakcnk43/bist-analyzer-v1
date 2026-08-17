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
from app.ml.adaptive_meta import AdaptiveMetaLearnerV4
from app.ml.transition_expert import RegimeTransitionExpert
from app.ml.monitoring import DriftGuardV2
from app.research.regimes import MarketRegimeDetector
from app.validation.evaluator import ResearchEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase21Research")

def main():
    logger.info("--- PHASE 21: ROBUST ADAPTIVE INTELLIGENCE & ONLINE META-LEARNING ---")

    # 1. Load Data (Reduced set for speed)
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:80]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            all_data.append(df)
    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)

    # 2. Setup Components
    evaluator = ResearchEvaluator()
    regime_detector = MarketRegimeDetector()
    quality_predictor = HighQualityMarketDayPredictor()
    meta_learner = AdaptiveMetaLearnerV4()
    transition_expert = RegimeTransitionExpert()
    drift_guard = DriftGuardV2()
    p12_scorer = ProductionScorer(MODELS_DIR)
    combo_opt = Top5CombinationOptimizer()

    # Intervals
    meta_train_end = '2026-05-01'
    test_df = global_df[global_df.index > meta_train_end].copy()
    test_dates = sorted(test_df.index.unique())

    # 3. Pre-train Transition Expert & Drift Reference
    logger.info("Initializing Intelligence Layer...")
    transition_expert.train(breadth_df[breadth_df.index <= meta_train_end],
                             pd.Series("NORMAL", index=breadth_df.index)) # Simple labels for research

    # 4. ONLINE ADAPTATION SIMULATION
    logger.info(f"Starting Online Adaptation Loop over {len(test_dates)} days...")

    results = []
    prediction_memory = {} # date -> picks

    for i, current_date in enumerate(test_dates):
        # A. Feedback Loop: Handle completed 5D outcomes from T-5
        if i >= 5:
            outcome_date = test_dates[i-5]
            if outcome_date in prediction_memory:
                picks = prediction_memory[outcome_date]
                actuals = global_df.loc[[outcome_date]].set_index('symbol_col')

                for _, p in picks.iterrows():
                    sym = p['symbol_col']
                    if sym in actuals.index:
                        hit = int(actuals.loc[sym]['target_return_5d'] > 0)
                        # Update Meta Learner with real outcome
                        meta_learner.record_outcome("general", outcome_date, hit)

        # B. Predict Step (T)
        day_df = test_df.loc[[current_date]].copy()
        day_mkt = breadth_df.loc[[current_date]]
        regime = regime_detector.detect_regime(day_mkt)

        # Contextual features
        trans_prob = transition_expert.predict_transition_prob(day_mkt)
        mkt_quality = quality_predictor.calculate_quality_score(day_mkt.iloc[0], regime)
        rel_feats = meta_learner.get_reliability_features(current_date)

        # Expert Probabilities
        scores = p12_scorer.calculate_production_scores(day_df, {'regime': regime})
        day_scored = day_df.join(scores)

        # Meta-Weighting (Adaptive V4)
        # In this simulation, we'll use the General Alpha adjusted by reliability
        trust_coeff = rel_feats.get('rel_general_20d', 0.5) / 0.5
        day_scored['production_alpha'] = (day_scored['production_alpha'] * trust_coeff).clip(0, 1)

        # Selection
        top5 = combo_opt.select_optimal_set(day_scored, k=5, regime=regime)
        prediction_memory[current_date] = top5

        # Calculate immediate metrics if available for scoreboard
        h = 0
        if not top5.empty:
            actuals = global_df.loc[[current_date]].set_index('symbol_col')
            valid = [s for s in top5['symbol_col'].tolist() if s in actuals.index]
            if valid:
                h = (actuals.loc[valid]['target_return_5d'] > 0).sum()

        results.append({
            'date': current_date,
            'hits': h,
            'trans_prob': trans_prob,
            'quality': mkt_quality,
            'trust': trust_coeff
        })

    res_df = pd.DataFrame(results)

    print("\n--- PHASE 21 FINAL SCOREBOARD (Online Simulated) ---")
    print(f"Adaptive Top5_3PLUS: {(res_df['hits'] >= 3).mean():.2%}")
    print(f"Mean Reliability Adjustment: {res_df['trust'].mean():.2f}")
    print(f"Max Transition Prob Detected: {res_df['trans_prob'].max():.2%}")

    res_df.to_csv(DATA_REPORTS_DIR / "PHASE21_DAILY_RESULTS.csv")

if __name__ == "__main__":
    main()
