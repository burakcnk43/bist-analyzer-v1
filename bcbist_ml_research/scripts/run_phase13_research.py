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
from app.features.boxes import DarvasBoxEngine
from app.ml.box_specialists import BoxExpert, BoxFailurePredictor
from app.ml.meta_optimizer import MetaGaterV3
from app.ml.specialists import SectorExpertManager, RegimeExpertManager
from app.research.regimes import MarketRegimeDetector, RegimeTransitionModel
from app.ml.production_scorer import ProductionScorer
from app.ml.selection import Top5CombinationOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase13Research")

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
    logger.info("--- PHASE 13: BOX THEORY & TOP5_3PLUS OPTIMIZATION ---")

    # 1. Load Data
    all_data_list = []
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')
    valid_list = valid_syms['symbol'].tolist()
    box_engine = DarvasBoxEngine()

    for symbol in valid_list[:200]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df.columns = [c.lower().strip() for c in df.columns]
            df['symbol_col'] = symbol
            df = box_engine.add_box_features(df)
            all_data_list.append(df)

    global_df = pd.concat(all_data_list).sort_index()
    logger.info(f"Loaded {len(global_df)} rows with Box features.")

    train_end = '2026-04-01'
    meta_end = '2026-06-01'
    test_end = '2026-08-05'

    train_df = global_df[global_df.index <= train_end]
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_end)]
    test_df = global_df[(global_df.index > meta_end) & (global_df.index <= test_end)]

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'box_', 'breakout_', 'is_breakout', 'pos_in_box']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    # 2. Train Specialists
    logger.info("Training Specialists...")
    sector_mgr = SectorExpertManager(MODELS_DIR, min_samples=500)
    sector_mgr.train_experts(train_df, selected_features, 'target_label_5d')

    regime_mgr = RegimeExpertManager(MODELS_DIR)
    detector = MarketRegimeDetector()
    train_df = train_df.copy()
    train_dates = train_df.index.unique()
    regime_map = {d: detector.detect_regime(train_df.loc[[d]]) for d in train_dates}
    train_df['regime'] = train_df.index.map(regime_map)
    regime_mgr.train_experts(train_df, selected_features, 'target_label_5d')

    box_expert = BoxExpert()
    box_expert.train(train_df, 'target_label_5d')

    # 3. Train MetaGaterV3
    logger.info("Training MetaGaterV3...")
    trans_mdl = RegimeTransitionModel()
    trans_mdl.train(train_df)

    p13_scorer = ProductionScorer(MODELS_DIR)
    base_ens = p13_scorer.models.get(5)

    base_features = []
    if base_ens:
        if hasattr(base_ens, 'classifier'): base_features.extend(base_ens.classifier.selected_features)
        if hasattr(base_ens, 'ranker'): base_features.extend(base_ens.ranker.selected_features)
    base_features = list(set(base_features))

    meta_records = []
    meta_dates = meta_df.index.unique()

    for date in meta_dates:
        day_df_raw = meta_df.loc[[date]]
        day_df = day_df_raw.copy()
        day_df.columns = [c.lower().strip() for c in day_df.columns]
        regime = detector.detect_regime(day_df)
        trans_p = trans_mdl.predict_transition_prob(day_df)

        try:
            mkt_ctx = {'regime': regime, 'transition_prob': trans_p}
            day_p12_scores = p13_scorer.calculate_production_scores(day_df_raw, mkt_ctx, sector_expert=sector_mgr, regime_expert=regime_mgr)
            p_gen = day_p12_scores['production_alpha'].values
        except:
            p_gen = np.zeros(len(day_df))

        probs = {
            'general': p_gen,
            'sector': sector_mgr.predict(day_df, base_ens),
            'regime': regime_mgr.predict(align_features(day_df, selected_features), regime, base_ens),
            'box': box_expert.predict_probs(day_df)
        }

        gater = MetaGaterV3()
        y = (day_df['target_return_5d'] > 0).astype(int)
        box_cols = [c for c in day_df.columns if 'box_' in c or 'is_breakout' in c]
        meta_features = gater.prepare_gating_features(day_df, probs, regime, trans_p, {}, day_df[box_cols])
        meta_features['is_hit'] = y.values
        meta_records.append(meta_features)

    full_meta_train = pd.concat(meta_records)
    final_gater = MetaGaterV3()
    final_gater.train(full_meta_train.drop(columns=['is_hit']), full_meta_train['is_hit'])

    # 4. Final Evaluation
    logger.info("Evaluating...")
    combo_opt = Top5CombinationOptimizer()
    test_dates = test_df.index.unique().sort_values()
    results = []

    for date in test_dates:
        day_df_raw = test_df.loc[[date]].copy()
        day_df_aligned = align_features(day_df_raw, list(set(base_features + selected_features)))
        day_df_aligned['sector'] = day_df_raw['sector'].values
        day_df_aligned['symbol_col'] = day_df_raw['symbol_col'].values

        mkt_ctx = {'regime': detector.detect_regime(day_df_raw), 'transition_prob': trans_mdl.predict_transition_prob(day_df_raw)}

        p12_scores = p13_scorer.calculate_production_scores(day_df_aligned, mkt_ctx, sector_expert=sector_mgr, regime_expert=regime_mgr)
        top5_12 = p13_scorer.select_top_5(p12_scores, day_df_aligned[['symbol_col', 'sector']])

        p13_scores = p13_scorer.calculate_production_scores(day_df_aligned, mkt_ctx, sector_expert=sector_mgr, regime_expert=regime_mgr, box_expert=box_expert, meta_v3=final_gater)
        top5_13 = p13_scorer.select_top_5(p13_scores, day_df_aligned[['symbol_col', 'sector']], optimizer=combo_opt)

        actuals = global_df.loc[[date]].set_index('symbol_col')
        h12 = (actuals.loc[top5_12['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_12.empty else 0
        h13 = (actuals.loc[top5_13['symbol_col'].tolist()]['target_return_5d'] > 0).sum() if not top5_13.empty else 0

        results.append({
            "date": date, "p12_hits": h12, "p13_hits": h13,
            "p13_ret": actuals.loc[top5_13['symbol_col'].tolist()]['target_return_5d'].mean() if not top5_13.empty else 0
        })

    res_df = pd.DataFrame(results)
    print(f"\nPHASE 12 Top5_3PLUS = {(res_df['p12_hits'] >= 3).mean():.2%}")
    print(f"PHASE 13 Top5_3PLUS = {(res_df['p13_hits'] >= 3).mean():.2%}")
    print(f"DELTA = {(res_df['p13_hits'] >= 3).mean() - (res_df['p12_hits'] >= 3).mean():+.2%}")

if __name__ == "__main__":
    main()
