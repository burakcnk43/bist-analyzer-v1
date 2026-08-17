import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProductionBacktest")

def main():
    logger.info("--- PHASE 6: DAILY PRODUCTION SIMULATION (BACKTEST) ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 5 # Production horizon
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'
    realized_ret_1d = 'target_return_1d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # 2. Train Ensemble
    logger.info("Training Production Ensemble...")
    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    tr_cls_all = research_df[selected_features + [target_cls]].dropna()
    p_cls.fit(tr_cls_all[selected_features], tr_cls_all[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    tr_rnk_all = research_df[selected_features + [target_rnk]].dropna()
    p_rnk.fit(tr_rnk_all[selected_features], tr_rnk_all[target_rnk])

    ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.4, w_rnk=0.6)

    # 3. Simulate Daily Trading
    test_dates = sorted(holdout_df.index.unique())
    portfolio_history = []
    transaction_cost = 0.001 # 10 bps

    logger.info(f"Simulating {len(test_dates)} trading days...")

    for date in test_dates:
        # Use .loc[[date]] to ALWAYS get a DataFrame even if 1 row
        day_data = holdout_df.loc[[date]].copy()

        if day_data.empty or realized_ret_1d not in day_data.columns: continue

        # Filter stocks that have features ready
        day_clean = day_data.dropna(subset=selected_features)
        if len(day_clean) < 10: continue

        # Predict Alpha Score
        day_clean['alpha'] = ensemble.get_alpha_score(day_clean[selected_features])

        # Select Top 5
        top_5 = day_clean.nlargest(5, 'alpha')

        # Realized performance (following day)
        # Note: target_return_1d is return from T to T+1
        avg_ret = top_5[realized_ret_1d].mean()
        net_ret = avg_ret - transaction_cost if not pd.isna(avg_ret) else 0

        portfolio_history.append({
            "date": date,
            "net_return": net_ret,
            "top_stocks": top_5['symbol_col'].tolist(),
            "hit_count": (top_5[realized_ret_1d] > 0).sum()
        })

    if not portfolio_history:
        logger.error("No trading days simulated.")
        return

    perf_df = pd.DataFrame(portfolio_history)
    perf_df['cum_return'] = (1 + perf_df['net_return']).cumprod()

    final_cum = perf_df['cum_return'].iloc[-1] - 1

    std = perf_df['net_return'].std()
    sharpe = (perf_df['net_return'].mean() / std * np.sqrt(252)) if std > 0 else 0

    print("\nProduction Backtest Summary (Top-5 1D Execution):")
    print(f"Total Period:      {len(perf_df)} days")
    print(f"Cumulative Return: {final_cum:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Avg Hit Count:     {perf_df['hit_count'].mean():.2f} / 5")

    perf_df.to_csv(DATA_REPORTS_DIR / "PHASE6_DAILY_TOP5_BACKTEST.csv", index=False)

    # Update Research Report with Backtest findings
    update_p6_report(final_cum, sharpe, perf_df['hit_count'].mean())

def update_p6_report(cum_ret, sharpe, avg_hits):
    path = DATA_REPORTS_DIR / "PHASE6_RANKING_RESEARCH_REPORT.md"
    if os.path.exists(path):
        with open(path, "a") as f:
            f.write("\n## Production Backtest (1D Execution)\n")
            f.write(f"- **Cumulative Return (180d)**: {cum_ret:.2%}\n")
            f.write(f"- **Sharpe Ratio**: {sharpe:.2f}\n")
            f.write(f"- **Avg Daily Hit Count**: {avg_hits:.2f} / 5\n")

if __name__ == "__main__":
    main()
