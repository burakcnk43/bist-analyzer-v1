import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.validation.walk_forward import WalkForwardValidation, purge_overlap
from app.ml.pipeline import ResearchPipeline
from app.research.backtest_engine import EconomicBacktester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase4Research")

def main():
    logger.info("--- PHASE 4: FULL RESEARCH CYCLE ---")

    # 1. Load All Features
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            df['sector_col'] = entry['sector']
            all_data_list.append(df)

    global_df = pd.concat(all_data_list).sort_index()

    # 2. Define Strict Holdout (Last 6 Months)
    last_date = global_df.index.max()
    holdout_start = last_date - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    horizons = [1, 5, 20]
    results = []

    # Pre-defined groups
    groups = {
        "Technical": ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_'],
        "Volume": ['volume', 'obv'],
        "Sector": ['rel_sector', 'sector_mean'],
        "Macro": ['macro_'],
        "KAP": ['event_', 'days_since']
    }

    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    all_target_cols = [c for c in global_df.columns if c.startswith('target_')]

    backtester = EconomicBacktester()

    # 3. Execution Loop
    for h in horizons:
        logger.info(f"Researching Horizon: {h}d")
        target_cls = f'target_up_{h}d'
        target_ret = f'target_return_{h}d'

        # Test configurations
        configs = [
            ("Technical Only", ["Technical"]),
            ("FULL MODEL", list(groups.keys()))
        ]

        for config_name, active_groups in configs:
            raw_features = []
            for g in active_groups:
                raw_features.extend([c for c in global_df.columns if any(p in c for p in groups[g])
                                    and c not in drop_cols and c not in all_target_cols])

            selected_features = sorted(list(set(raw_features)))
            logger.info(f"  Config: {config_name}")

            wf = WalkForwardValidation(n_folds=3, train_size_days=252*2, test_size_days=120, purge_days=20)
            splits = wf.split(research_df)

            fold_accs = []
            strategy_cum_rets = []

            for fold, (train_df, test_df) in enumerate(splits):
                pipeline = ResearchPipeline(model_type='xgboost', n_features=30)

                # Filter for this fold
                train_fold_data = train_df[selected_features + [target_cls]].dropna()
                X_train = train_fold_data[selected_features]
                y_train = train_fold_data[target_cls]

                if len(X_train) < 500: continue
                pipeline.fit(X_train, y_train)

                # Evaluate Prediction
                test_fold_data = test_df[selected_features + [target_cls, target_ret]].dropna()
                if test_fold_data.empty: continue
                X_test = test_fold_data[selected_features]
                y_test = test_fold_data[target_cls]

                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)

                acc = accuracy_score(y_test, y_pred)
                fold_accs.append(acc)

                # Run Economic Backtest for this fold
                bt_metrics = backtester.run_simple_backtest(test_fold_data, y_prob, horizon=h)
                strategy_cum_rets.append(bt_metrics.get('cumulative_return', 0))

            # Final Evaluation on Holdout (Only for FULL MODEL)
            holdout_acc = 0
            if config_name == "FULL MODEL":
                pipeline = ResearchPipeline(model_type='xgboost', n_features=30)
                # Use all research data to train for final holdout
                train_holdout = research_df[selected_features + [target_cls]].dropna()
                pipeline.fit(train_holdout[selected_features], train_holdout[target_cls])

                eval_holdout = holdout_df[selected_features + [target_cls]].dropna()
                if not eval_holdout.empty:
                    y_holdout_pred = pipeline.predict(eval_holdout[selected_features])
                    holdout_acc = accuracy_score(eval_holdout[target_cls], y_holdout_pred)

            results.append({
                "Horizon": h,
                "Config": config_name,
                "Avg_Accuracy": np.mean(fold_accs) if fold_accs else 0,
                "Holdout_Accuracy": holdout_acc,
                "Avg_Strategy_Ret": np.mean(strategy_cum_rets) if strategy_cum_rets else 0
            })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(DATA_REPORTS_DIR / "PHASE4_FULL_RESEARCH_RESULTS.csv", index=False)

    # 4. Generate Markdown Report
    generate_p4_report(summary_df)

def generate_p4_report(summary_df):
    report = "# BCBIST Phase 4 Full Research Report\n\n"
    report += "## Executive Summary\n"
    report += f"Establish Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    report += f"Data Range: 5 Years\n\n"

    report += "## Performance Table\n"
    report += summary_df.to_markdown(index=False)

    report += "\n\n## Insights\n"
    best_row = summary_df.loc[summary_df['Avg_Accuracy'].idxmax()]
    report += f"- **Best Accuracy**: {best_row['Avg_Accuracy']:.2%} achieved on {best_row['Horizon']}d horizon with {best_row['Config']}.\n"

    with open(DATA_REPORTS_DIR / "PHASE4_FULL_RESEARCH_REPORT.md", "w") as f:
        f.write(report)
    logger.info("Markdown report generated.")

if __name__ == "__main__":
    main()
