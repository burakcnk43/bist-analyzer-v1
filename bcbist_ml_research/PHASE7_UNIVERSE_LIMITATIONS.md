# Phase 7: Universe & Survivorship Bias Limitations

This document outlines the known limitations regarding the Borsa Istanbul (BIST) universe expansion and historical data integrity.

## 1. Survivorship Bias
- **Current Universe**: The research uses the BIST universe constituents as of August 2026.
- **Limitation**: Historical training data (2021-2026) only includes companies that were still listed and active in August 2026. 
- **Impact**: Companies that went bankrupt, were delisted, or merged out of existence during the 5-year period are not represented in the training set. This typically results in an upward bias in historical performance metrics ("Survivorship Bias").

## 2. Listing & IPO Gaps
- **IPO Period**: Many BIST companies (e.g., `ASTOR.IS`, `REEDR.IS`, `CWENE.IS`) had their IPOs between 2023 and 2025.
- **Limitation**: These companies have much shorter historical lookbacks than established blue chips. 
- **Handling**: The model handles these as "Missing" during early periods. However, the abundance of new high-growth stocks in the latter half of the dataset may skew the model's perception of "typical" stock behavior.

## 3. Index Membership (PIT)
- **BIST 100/50/30 Boundaries**: Real-world index membership changes quarterly. 
- **Limitation**: We do not currently possess a Point-in-Time (PIT) database of exact index constituents for every historical day.
- **Handling**: We use cross-sectional liquidity (Median Turnover) as a proxy for index eligibility.

## 4. Ticker Changes
- **Identity Shifts**: Some Turkish companies change tickers during rebranding.
- **Limitation**: If the Yahoo Finance mapping for an old ticker is broken, that historical period is lost to the model.

## Final Disclaimer
The high performance reported in Phase 7 (**60.8% Hit Rate**) should be interpreted with these biases in mind. Real-world "live" performance on newly listed or declining companies may differ from these retrospective results.
