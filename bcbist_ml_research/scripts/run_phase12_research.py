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
from app.ml.specialists import SectorExpertManager, RegimeExpertManager
from app.ml.meta_optimizer import MetaGaterV3
from app.research.regimes import MarketRegimeDetector, RegimeTransitionModel
from app.ml.production_scorer import ProductionScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase12Research")

def main():
    logger.info("--- PHASE 12: SPECIALIST ENSEMBLE RESEARCH ---")

    # 1. Load Data
    all_data_list = []
    # Use more symbols for faster research
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')
    valid_list = valid_syms['symbol'].tolist()

    # Use entire valid universe for production-grade evaluation
    for symbol in valid_list:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)

    global_df = pd.concat(all_data_list).sort_index()
    logger.info(f"Loaded {len(global_df)} rows for {len(all_data_list)} symbols.")

    # Split:
    # Training (Frozen): Up to 2026-04-01 (Broaden for specialist samples)
    # Meta-Training (OOS): 2026-04-02 to 2026-06-01
    # Final Validation (Forward): 2026-06-02 to 2026-08-05
    train_end = '2026-04-01'
    meta_end = '2026-06-01'
    test_end = '2026-08-05'

    train_df = global_df[global_df.index <= train_end]
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_end)]
    test_df = global_df[(global_df.index > meta_end) & (global_df.index <= test_end)]

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    # 2. Train Specialists
    logger.info("Training Sector Specialists...")
    sector_mgr = SectorExpertManager(MODELS_DIR, min_samples=500)
    sector_mgr.train_experts(train_df, selected_features, 'target_label_5d')

    logger.info("Training Regime Specialists...")
    regime_mgr = RegimeExpertManager(MODELS_DIR)
    detector = MarketRegimeDetector()
    train_df = train_df.copy()

    train_dates = train_df.index.unique()
    regime_map = {d: detector.detect_regime(train_df.loc[[d]]) for d in train_dates}
    train_df['regime'] = train_df.index.map(regime_map)

    regime_mgr.train_experts(train_df, selected_features, 'target_label_5d')

    # 3. Train Transition Model
    logger.info("Training Regime Transition Model...")
    trans_mdl = RegimeTransitionModel()
    trans_mdl.train(train_df)

    # 4. Train MetaGaterV3
    logger.info("Training MetaGaterV3 (Gating Specialists)...")
    p12_scorer = ProductionScorer(MODELS_DIR)
    # Base model for proposals (Production 5D Ensemble)
    base_ens = p12_scorer.models.get(5)
    if not base_ens:
        logger.warning("Base 5D model not found in scorer. Loading directly.")
        base_ens = joblib.load(MODELS_DIR / "phase10_cls_5d.joblib")

    # Load Phase 11 Meta for baseline comparison
    p11_meta = None
    if (MODELS_DIR / "phase11_meta_v2.joblib").exists():
        p11_meta = joblib.load(MODELS_DIR / "phase11_meta_v2.joblib")

    meta_records = []
    meta_dates = meta_df.index.unique()

    for date in meta_dates:
        day_df = meta_df.loc[[date]]
        regime = detector.detect_regime(day_df)
        trans_p = trans_mdl.predict_transition_prob(day_df)

        # Specialist Proposals
        probs = {
            'general': base_ens.get_alpha_score(day_df[selected_features]),
            'sector': sector_mgr.predict(day_df, base_ens),
            'regime': regime_mgr.predict(day_df[selected_features], regime, base_ens)
        }

        gater = MetaGaterV3()
        y = (day_df['target_return_5d'] > 0).astype(int)

        meta_features = gater.prepare_gating_features(day_df, probs, regime, trans_p, {})
        meta_features['is_hit'] = y.values
        meta_records.append(meta_features)

    full_meta_train = pd.concat(meta_records)
    final_gater = MetaGaterV3()
    final_gater.train(full_meta_train.drop(columns=['is_hit']), full_meta_train['is_hit'])
    joblib.dump(final_gater, MODELS_DIR / "phase12_meta_v3.joblib")

    # 5. Final Evaluation
    logger.info("Running Side-by-Side Comparison on Validation Period...")

    test_dates = test_df.index.unique().sort_values()
    results = []

    for date in test_dates:
        day_df = test_df.loc[[date]].copy()
        mkt_ctx = {
            'regime': detector.detect_regime(day_df),
            'transition_prob': trans_mdl.predict_transition_prob(day_df),
            'reliability': {}
        }

        # P11 Scorer (Actual Phase 11 Logic)
        p11_scores = p12_scorer.calculate_production_scores(
            day_df[selected_features],
            mkt_ctx,
            meta_v2=p11_meta
        )
        top5_11 = p12_scorer.select_top_5(p11_scores, day_df[['symbol_col', 'sector']])

        # P12 Scorer (Full Specialists + Gater)
        p12_scores = p12_scorer.calculate_production_scores(
            day_df[selected_features + ['sector']],
            mkt_ctx,
            sector_expert=sector_mgr,
            regime_expert=regime_mgr,
            meta_v3=final_gater
        )
        top5_12 = p12_scorer.select_top_5(p12_scores, day_df[['symbol_col', 'sector']])

        actuals = global_df.loc[[date]].set_index('symbol_col')

        h11 = (actuals.loc[top5_11['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_11.empty else 0
        h12 = (actuals.loc[top5_12['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_12.empty else 0

        results.append({
            "date": date,
            "p11_hits": h11,
            "p12_hits": h12,
            "p11_ret": actuals.loc[top5_11['symbol_col'].tolist()]['target_return_5d'].mean() if not top5_11.empty else 0,
            "p12_ret": actuals.loc[top5_12['symbol_col'].tolist()]['target_return_5d'].mean() if not top5_12.empty else 0
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE12_FINAL_COMPARISON.csv", index=False)

    print("\n" + "="*50)
    print("PHASE 12 FINAL SCOREBOARD")
    print("="*50)
    print(f"PHASE 11 Top5_2PLUS = {(res_df['p11_hits'] >= 2).mean():.2%}")
    print(f"PHASE 12 Top5_2PLUS = {(res_df['p12_hits'] >= 2).mean():.2%}")
    print(f"DELTA (2+) = {(res_df['p12_hits'] >= 2).mean() - (res_df['p11_hits'] >= 2).mean():+.2%}")
    print("-" * 30)
    print(f"PHASE 11 Top5_1PLUS = {(res_df['p11_hits'] >= 1).mean():.2%}")
    print(f"PHASE 12 Top5_1PLUS = {(res_df['p12_hits'] >= 1).mean():.2%}")
    print(f"PHASE 11 Avg Return = {res_df['p11_ret'].mean():.2%}")
    print(f"PHASE 12 Avg Return = {res_df['p12_ret'].mean():.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
