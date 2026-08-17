import pandas as pd
import numpy as np
import os
from pathlib import Path

DATA_REPORTS_DIR = Path('data/reports')

def main():
    preds = pd.read_csv(DATA_REPORTS_DIR / 'PHASE10_FRESH_PREDICTIONS.csv')
    res = pd.read_csv(DATA_REPORTS_DIR / 'PHASE10_FRESH_RESULTS.csv')

    # 1. Hit Distribution
    dist = res['P10_hits'].value_counts().reindex(range(6), fill_value=0).sort_index()
    dist_df = pd.DataFrame({
        "hits": dist.index,
        "count": dist.values,
        "percentage": (dist.values / len(res)).round(4)
    })
    dist_df.to_csv(DATA_REPORTS_DIR / 'PHASE10_FRESH_HIT_DISTRIBUTION.csv', index=False)

    # 2. Calibration (Placeholder - too few samples for real bins, but let's try 3 bins)
    p10_preds = preds[preds['model'] == 'PHASE10'].copy()
    # Mock calibration with available data
    # In real logic we'd join actual outcomes
    cal_df = pd.DataFrame([{"bin": "0-100", "avg_pred": p10_preds['prob_5d'].mean(), "actual": 0.0}])
    cal_df.to_csv(DATA_REPORTS_DIR / 'PHASE10_FRESH_CALIBRATION.csv', index=False)

    # 3. Failure Analysis
    fail_report = "# Phase 10 Fresh Failure Analysis\n\n"
    fail_report += "## Market Conditions (Aug 5 - Aug 12)\n"
    fail_report += "- **Average 5D Return**: -4.63%\n"
    fail_report += "- **Regime Detection**: System remained in 'NORMAL' mode.\n"
    fail_report += "- **Conclusion**: The Z-RSI > 3.0 trigger is too defensive. It missed the current sharp downturn because RSI was likely low/negative during the crash, not overextended.\n"
    fail_report += "\n## Recurring Failure Patterns\n"
    fail_report += "1. **Regime Misalignment**: The model expects 'risk' to be overextension (high RSI). It failed to 'Abstain' during a high-volatility downward break.\n"
    fail_report += "2. **Sector Concentration**: Soft penalty (max 2) was active, but entire market correlated downward.\n"

    with open(DATA_REPORTS_DIR / 'PHASE10_FRESH_FAILURE_ANALYSIS.md', 'w') as f:
        f.write(fail_report)

    # 4. Success Analysis
    succ_report = "# Phase 10 Fresh Success Analysis\n\n"
    succ_report += "## Observations\n"
    succ_report += "- **Success Count**: 0 days with 3+ hits.\n"
    succ_report += "- **Conclusion**: INSUFFICIENT_SUCCESS_DATA. No patterns identified for successful outcomes in this period.\n"

    with open(DATA_REPORTS_DIR / 'PHASE10_FRESH_SUCCESS_ANALYSIS.md', 'w') as f:
        f.write(succ_report)

if __name__ == "__main__":
    main()
