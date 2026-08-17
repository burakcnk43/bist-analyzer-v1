import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, MODELS_DIR
from app.ml.group_failure import GroupOutcomePredictorV2
from app.ml.transition_expert import RegimeTransitionExpert
from app.ml.meta_labeler import AlphaTrustModel
from app.ml.box_conditional import DarvasQualityScorer
from app.ml.event_chains import EventChainExpert
from app.ml.production_scorer import ProductionScorer
from app.ml.selection import Top5CombinationOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainProductionMeta")

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
    # 1. Load Data
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    all_data = []
    for symbol in valid_syms[:100]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data.append(df)
    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)

    # Intervals
    train_end = '2025-12-31'
    meta_train_end = '2026-05-01'
    meta_df = global_df[(global_df.index > train_end) & (global_df.index <= meta_train_end)].copy()

    prod_models_dir = MODELS_DIR / "production"
    prod_models_dir.mkdir(exist_ok=True)

    # 2. Train Regime Transition Expert
    transition_expert = RegimeTransitionExpert()
    regime_labels = pd.Series("SIDEWAYS", index=breadth_df.index)
    regime_labels.loc['2026-01-01':] = "BULL"
    transition_expert.train(breadth_df, regime_labels)
    transition_expert.save(prod_models_dir / "transition_expert.joblib")

    # 3. Train Specialists (AlphaTrust, Darvas, Event)
    logger.info("Training additional specialists...")
    trust_model = AlphaTrustModel()
    trust_model.train(meta_df, (meta_df['target_return_5d'] > 0))
    trust_model.save(prod_models_dir / "alpha_trust.joblib")

    darvas_scorer = DarvasQualityScorer()
    darvas_scorer.train_with_quality(meta_df, (meta_df['target_return_5d'] > 0))
    darvas_scorer.save(prod_models_dir / "darvas_quality.joblib")

    event_expert = EventChainExpert()
    event_expert.train(meta_df, (meta_df['target_return_5d'] > 0))
    event_expert.save(prod_models_dir / "event_chain.joblib")

    # 4. Train Group Outcome Predictor
    scorer = ProductionScorer(MODELS_DIR)
    optimizer = Top5CombinationOptimizer()
    group_records = []

    meta_dates = sorted(meta_df.index.unique())
    for date in meta_dates:
        day_data = meta_df.loc[[date]]
        day_mkt = breadth_df.loc[[date]]

        scores = scorer.calculate_production_scores(day_data, {'regime': 'NORMAL'})
        day_full = day_data.join(scores)
        top5 = optimizer.select_optimal_set(day_full, k=5)

        actuals = global_df.loc[[date]].set_index('symbol_col')
        valid = [s for s in top5['symbol_col'].tolist() if s in actuals.index]
        hits = (actuals.loc[valid]['target_return_5d'] > 0).sum() if valid else 0

        predictor = GroupOutcomePredictorV2()
        feats = predictor.prepare_group_features(day_mkt.iloc[0], day_full)
        feats['hit_count'] = hits
        group_records.append(feats)

    group_history = pd.concat(group_records)
    outcome_predictor = GroupOutcomePredictorV2()
    outcome_predictor.train(group_history.drop(columns=['hit_count']), group_history['hit_count'])
    outcome_predictor.save(prod_models_dir / "group_outcome_v2.joblib")

    logger.info("Production meta-models trained and saved.")

if __name__ == "__main__":
    main()
