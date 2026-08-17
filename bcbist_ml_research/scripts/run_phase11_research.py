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
from app.ml.production_scorer import ProductionScorer
from app.research.regimes import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase11Research")

def bootstrap_metric(values, n_iter=10000):
    stats = []
    for _ in range(n_iter):
        sample = np.random.choice(values, size=len(values), replace=True)
        stats.append(np.mean(sample))
    return np.percentile(stats, [2.5, 97.5])

def main():
    logger.info("--- PHASE 11: ADAPTIVE SELF-LEARNING VALIDATION ---")

    # 1. Load Data (Focus on last 180 days for final validation)
    all_data_list = []
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    for symbol in valid_syms[:150]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            df['sector_col'] = 'Other' # For demo
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    test_start = global_df.index.max() - timedelta(days=120)
    test_df = global_df[global_df.index >= test_start]

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    # 2. Load Frozen Assets
    detector = MarketRegimeDetector()
    p11_scorer = ProductionScorer(MODELS_DIR)
    meta_v2 = joblib.load(MODELS_DIR / "phase11_meta_v2.joblib")

    # 3. Simulation Loop
    dates = test_df.index.unique().sort_values()
    results = []

    for date in dates:
        day_df = test_df.loc[[date]]
        mkt_ctx = {'regime': detector.detect_regime(day_df)}

        # P10 Baseline (No Meta V2)
        p10_res = p11_scorer.calculate_production_scores(day_df[selected_features], mkt_ctx, meta_v2=None)
        top5_10 = p11_scorer.select_top_5(p10_res, day_df[['symbol_col', 'sector_col']])

        # P11 Adaptive (With Meta V2)
        p11_res = p11_scorer.calculate_production_scores(day_df[selected_features], mkt_ctx, meta_v2=meta_v2)
        top5_11 = p11_scorer.select_top_5(p11_res, day_df[['symbol_col', 'sector_col']])

        day_full = global_df.loc[[date]] if not isinstance(global_df.loc[date], pd.DataFrame) else global_df.loc[date]
        day_actuals = day_full.set_index('symbol_col')

        # Hits P10
        if not top5_10.empty:
            actuals_10 = day_actuals.loc[top5_10['symbol_col'].tolist()]
            h10 = (actuals_10['target_return_5d'] > 0).sum()
        else: h10 = 0

        # Hits P11
        if not top5_11.empty:
            actuals_11 = day_actuals.loc[top5_11['symbol_col'].tolist()]
            h11 = (actuals_11['target_return_5d'] > 0).sum()
            r11 = actuals_11['target_return_5d'].mean()
        else: h11, r11 = 0, 0

        results.append({
            "date": date,
            "p10_hits": h10, "p10_3plus": 1 if h10 >= 3 else 0,
            "p11_hits": h11, "p11_3plus": 1 if h11 >= 3 else 0,
            "p11_ret": r11
        })

    res_df = pd.DataFrame(results)

    # 4. Statistical Analysis
    p11_3p_ci = bootstrap_metric(res_df['p11_3plus'].values)
    diff_values = res_df['p11_3plus'] - res_df['p10_3plus']
    diff_ci = bootstrap_metric(diff_values.values)

    # 5. Scoreboard
    print("\n" + "="*50)
    print("PHASE 11 FINAL SCOREBOARD (ADAPTIVE OOS)")
    print("="*50)
    print(f"P10 Top5_3PLUS = {res_df['p10_3plus'].mean():.2%}")
    print(f"P11 Top5_3PLUS = {res_df['p11_3plus'].mean():.2%}")
    print(f"Improvement   = {res_df['p11_3plus'].mean() - res_df['p10_3plus'].mean():+.2%}")
    print(f"95% CI (Delta) = [{diff_ci[0]:+.2%}, {diff_ci[1]:+.2%}]")

    if diff_ci[0] > 0:
        verdict = "PHASE11_SIGNIFICANT_IMPROVEMENT"
    elif res_df['p11_3plus'].mean() > res_df['p10_3plus'].mean():
        verdict = "PHASE11_PROMISING"
    else:
        verdict = "PHASE11_NO_SIGNIFICANT_IMPROVEMENT"

    print(f"FINAL VERDICT = {verdict}")
    print("="*50)

    # 6. Save results
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE11_FINAL_COMPARISON.csv", index=False)

if __name__ == "__main__":
    main()
