
# PHASE 3 RESEARCH REPORT - REAL COMPANY-SPECIFIC KAP ANALYSIS
**Generated on:** 2026-08-11T16:27:08.301488

## 1. Executive Summary
- **Total Real Company Events:** 23
- **KAP Status:** OPERATIONAL (CSV/Real)

## 2. Model Ablation (Alpha Contribution)
| Step | Configuration | Accuracy | Gain |
|------|---------------|----------|------|
| A: | A: Technical | 0.4812 | +0.0000 |
| B: | B: Tech + Vol | 0.4757 | -0.0056 |
| C: | C: + Sector | 0.4914 | +0.0157 |
| D: | D: + Macro | 0.5263 | +0.0349 |
| E: | E: + Fundamental | 0.5263 | +0.0000 |
| F: | F: + Macro Events | 0.5241 | -0.0023 |
| G: | G: FULL (Inc. Company KAP) | 0.5200 | -0.0040 |

## 3. High-Impact Event Types
Types associated with the strongest 5-day post-event moves.
| Event Type | Samples | Mean Return | Win Rate |
|------------|---------|-------------|----------|
| financial_results | 18 | 3.25% | 83.33% |
| central_bank | 576 | 1.04% | 52.60% |

## 4. Sector Sensitivity Ranking
Sectors responding most strongly to corporate disclosures.
| Sector | Samples | Mean Return | Win Rate |
|--------|---------|-------------|----------|
| Technology | 96 | 2.74% | 47.92% |
| Apparel | 52 | 2.59% | 67.31% |
| Polymers | 48 | 2.19% | 64.58% |
| REIT | 96 | 1.97% | 57.29% |
| Telecommunications | 99 | 1.71% | 64.65% |
| Construction | 99 | 1.70% | 59.60% |
| Energy | 192 | 1.67% | 52.60% |
| Beverages | 96 | 1.61% | 70.83% |
| Aviation | 150 | 1.57% | 57.33% |
| White Goods | 96 | 1.51% | 61.46% |
| Banking | 209 | 1.48% | 63.16% |
| Holding | 201 | 1.37% | 62.69% |
| Defense | 51 | 1.34% | 60.78% |
| Textile | 48 | 1.15% | 54.17% |
| Steel | 97 | 1.02% | 59.79% |
| Automotive | 195 | 0.92% | 51.28% |
| Retail | 152 | 0.82% | 56.58% |
| Petrochemicals | 99 | 0.74% | 57.58% |
| Catering | 48 | 0.68% | 60.42% |
| Cement | 96 | 0.43% | 50.00% |
| Agriculture | 48 | -0.04% | 43.75% |
| Insurance | 48 | -0.16% | 39.58% |
| Glass | 54 | -0.44% | 44.44% |

## 5. Technical + KAP Interaction
| Technical State | Has Event | Mean Return | Win Rate |
|-----------------|-----------|-------------|----------|
| Neutral | 0 | 0.36% | 50.20% |
| Neutral | 1 | 1.04% | 55.46% |
| Overbought | 0 | 0.70% | 52.71% |
| Overbought | 1 | 0.97% | 60.23% |
| Oversold | 0 | 0.95% | 54.01% |
| Oversold | 1 | 6.88% | 85.84% |

## 6. Conclusions
- **Alpha Decay:** We observe whether KAP impact is immediate (1D) or sustained (20D).
- **Synergy:** Technical indicators like RSI oversold show different win rates when combined with news.
