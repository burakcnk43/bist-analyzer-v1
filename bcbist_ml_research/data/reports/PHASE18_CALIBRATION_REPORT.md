# PHASE 18: CALIBRATION & INVERSION AUDIT

## The "Inversion" Observed
- **Problem**: `GroupOutcomePredictor` predicted high failure risk on high-success days.
- **Root Cause**: The model over-weighted `market_z_rsi_14`. In the validation period, high RSI (Overextension) was a predictor of CONTINUED momentum, but in the training period (Jan-May), it was a predictor of reversal.
- **Fix**: Phase 18 introduced `HighQualityMarketDayPredictor` which uses breadth momentum to override static RSI thresholds.

## Current Calibration (Brier Scores)
- Individual P(Success): 0.18
- Group P(3PLUS): 0.22 (Still needs work)
