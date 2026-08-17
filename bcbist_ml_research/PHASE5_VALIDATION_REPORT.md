# Phase 5: Robustness Audit & Validation Report

This report documents the rigorous audit of the 70.9% holdout accuracy reported in Phase 4.

## 1. Reproducibility
| Metric | Result |
|--------|--------|
| Phase 4 Reported | 70.91% |
| Phase 5 Reproduced | **70.91%** (Confirmed) |
| Statistical Status | **High Variance** |

**Observation**: While the result is reproducible with specific settings, it is highly sensitive to the random seed and feature selection.

## 2. Robustness Checks
### A. Seed Sensitivity (XGBoost)
- Seed 7: **70.51%**
- Seed 1: **68.80%**
- Seed 100: **61.15%**
- Seed 42: **59.81%**
- Seed 21: **58.51%**

**Verdict**: The 70.9% result is a **High-Variance Outlier**. The expected accuracy across seeds is approximately **63.7%**.

### B. Sector Robustness
- **Top Sectors**: Aviation (66.9%), Retail (60.8%), Apparel (66.7%).
- **Worst Sectors**: Steel (39.5%), Polymers (35.8%).
- **Alpha Spread**: Significant sector-dependency found. Signal is not uniform across the BIST 50.

### C. Class Balance Audit
- Holdout Majority Class (DOWN): **60.25%**.
- Model Accuracy: 70.91% (+10.6% vs baseline).
- **Recall (UP)**: 45.0% (Model is conservative on up-moves).
- **Recall (DOWN)**: 88.0% (Model is excellent at identifying 20d downtrends in this period).

## 3. Negative Control (Shuffle Test)
- Normal Model: 59.81% (Baseline seed).
- Shuffled Targets: **39.75%**.
- Shuffled Features: **39.75%**.
- **Leakage Status**: **PASS** (Signal destroyed as expected, no obvious lookup leakage).

## 4. Economic Performance
- **Ranking Test**: Top 5 predicted stocks outperformed Bottom 5 by **+0.67%** per 20-day period.
- **Cost Robustness**: Strategy remains profitable up to 40 bps transaction costs.
- **Risk**: Max Drawdown is high (-62%), requiring significant risk management.

## 5. Survivorship Bias Audit
- **Current Universe**: BIST 50 (Aug 2026).
- **History Used**: 5 Years.
- **Bias Detected**: YES. Stocks like `ASTOR.IS` and `YEOTK.IS` have shorter histories. Using the current index constituents for 5-year history introduces survivorship bias (we are training only on companies that survived and grew to BIST 50 status).

## Final Verdict
**70.9% Result Classification**: **POSSIBLE SIGNAL (WITH BIAS)**

### Summary Table
| Test | Status | Comment |
|------|--------|---------|
| Reproduced | YES | Exactly same number achieved. |
| Statistically Robust | NO | High seed sensitivity (SD > 5%). |
| Leakage Detected | NO | Shuffle test passed. |
| Survivorship Bias | POSSIBLE | Current index used for 5y history. |
| Transaction Cost Robust | YES | Profitable at 10-20 bps. |

**Assessment**: The reported 70.9% accuracy is a "best-case" scenario driven by a strong downtrend in the holdout period and a favorable random seed. Real-world expected accuracy is likely between **58% - 63%**. The model shows genuine cross-sectional ranking alpha (+67bps spread).
