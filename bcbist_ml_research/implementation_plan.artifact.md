# Implementation Plan - Phase 13: Darvas Box Specialist & Top5_3PLUS Optimization

This phase introduces Darvas Box Theory as a specialist research layer and implements a secondary optimization stage for Top-5 selection to maximize the probability of achieving 3+ successful picks per day.

## User Review Required

> [!IMPORTANT]
> **Strict Lookahead Controls**: Darvas boxes are traditionally defined by "waiting 3 days" to confirm a top/bottom. We must ensure that at any timestamp `T`, the box features ONLY use data from `T-k` to `T`.
> **Combination Optimization**: We are moving from "Best 5 Individual Scores" to "Best 5-Stock Portfolio". This considers the joint probability of success and inter-stock correlations.

## Open Questions
- For the "Box Top" definition, should we use a fixed 3-day wait (traditional Darvas) or a volatility-adaptive window? (Suggesting testing both in validation).
- Should "Breakout" be defined strictly on `close` or also on `high` wicks? (Planned to test both as separate features).

## Proposed Changes

### 1. Box Detection Engine
- [NEW] `app/features/boxes.py`:
    - `DarvasBoxEngine`: Implements rolling box identification.
    - Features: `box_top`, `box_bottom`, `box_age`, `box_tightness`, `box_staircase_score`.
    - Breakout Metrics: `breakout_strength`, `volume_confirmation_ratio`, `is_wick_breakout`.

### 2. Box Specialist Layer
- [NEW] `app/ml/box_specialists.py`:
    - `BoxExpert`: XGBoost model trained specifically on box-related features.
    - `BoxFailurePredictor`: Focuses on identifying "False Breakouts" (Bull traps).
    - `BoxContinuationPredictor`: Estimates the likelihood of momentum following a breakout.

### 3. Meta-Gating Upgrade
- [MODIFY] `app/ml/meta_optimizer.py`:
    - Upgrade `MetaGaterV3` to incorporate `BoxExpert` outputs.
    - New Inputs: Box Reliability (rolling OOS), Breakout Confidence, Regime-Box Fit.

### 4. Top-5 Portfolio Optimizer
- [NEW] `app/ml/selection.py`:
    - `Top5CombinationOptimizer`: Uses a second-stage learner to select a set of 5 stocks that maximizes `P(hits >= 3)`.
    - Correlation-Awareness: Penalizes clusters of highly correlated stocks to avoid 0/5 "wipeout" days.

### 5. Research & Validation
- [NEW] `scripts/run_phase13_research.py`:
    - Implements the 10-seed walk-forward loop.
    - Ablation study for Box stack vs Phase 12 baseline.
    - Success/Failure Day mining specifically for Darvas structures.

## Verification Plan

### Automated Tests
- Vectorized check for no-lookahead in `DarvasBoxEngine`.
- Statistical significance test (Bootstrap 10k) on `Top5_3PLUS` improvement.
- Evaluation of average returns and drawdown during "Staircase" vs "Broken Box" periods.

### Manual Verification
- Reviewing `PHASE13_BOX_FAILURE_ANALYSIS.md` to identify BIST-specific bull trap patterns.
- Checking `PHASE13_SECTOR_BOX_ANALYSIS.md` to see if certain sectors (e.g., Tech) obey Box Theory better than others.
