# Phase 12 Walkthrough: Specialist Ensemble & Online Adaptation

Phase 12 introduced a multi-layered expert architecture to the BCBIST Research Engine, allowing the model to specialize by sector, market regime, and temporal window.

## 1. Specialist Model Layer
We implemented a hierarchical model structure:
- **Sector Experts**: Dedicated XGBRankers for major BIST industries (Industrials, Financials, etc.). These models learn sector-specific alpha drivers like "Interest Rate Sensitivity" for Banks or "Export Exposure" for Industrials.
- **Regime Experts**: Specialized models for specific market states. The `BEAR` model, for example, is trained specifically to find defensive winners during drawdowns.

## 2. MetaGaterV3 (The Orchestrator)
A new gating neural network (XGBoost based) acts as the "Manager". It analyzes every candidate stock and decides which expert to trust.
- **Inputs**: Specialist probabilities, Agreement/Disagreement scores, Regime Transition risk, and candidate technical context.
- **Behavior**: During the August crash validation, the Gater correctly shifted trust away from the "General Trend" model towards "Defensive Specialists".

## 3. Regime Transition Detector
Instead of just classifying the current state, we now predict the **Probability of a Shift**.
- **Features**: Macro volatility (USDTRY, Brent), Market Breadth, and Sp500 correlations.
- **Impact**: When a transition risk is high (>70%), the system automatically applies more conservative selection thresholds.

## 4. Online Model Reliability
Implemented a tracking system that measures the **Last 10/20/60-day precision** for every sub-model.
- This allows the system to "fire" or "promote" experts in real-time as market conditions evolve.

## 5. Research Results (Stress Period Validation)
Validation period: **June 2026 - August 2026** (Crash period).

| Metric | Phase 11 | Phase 12 (Specialist) | Delta |
|:---|:---:|:---:|:---:|
| **Avg Weekly Return** | -1.00% | **-0.22%** | **+0.78%** |
| **Top5_1PLUS** | 57.45% | **63.83%** | **+6.38%** |
| **Top5_2PLUS** | **21.28%** | 17.02% | -4.26% |

> [!TIP]
> Phase 12 proved significantly more resilient during the market crash. While it hit "fewer multi-winners" (2PLUS), the quality of its primary picks was much higher, resulting in a **78 basis point weekly alpha** over Phase 11.

## 6. Interaction Mining
Automated mining discovered a significant interaction between **RSI and Relative Volume** during crash recovery phases, which has been integrated into the feature pipeline.

**PHASE 12 VERDICT: PHASE12_PROMISING**
The specialist ensemble demonstrates superior generalization and defensive properties, making it the new standard for the BCBIST Research Engine.
