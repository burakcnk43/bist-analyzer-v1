
# PHASE 2 RESEARCH REPORT - BCBIST ML Research Engine (KAP & Events)
**Generated on:** 2026-08-11T14:16:29.182648

## 1. Executive Summary
- **Stocks Analyzed:** 50
- **Real Events Collected:** 12
- **KAP Data Availability:** LIMITED/UNAVAILABLE

## 2. Feature Ablation Results
How much value each feature set adds to directional accuracy (5d).

| Configuration | Features | Avg Accuracy |
|---------------|----------|--------------|
| A: Technical Only | 31 | 0.4812 |
| B: Tech + Vol | 39 | 0.4757 |
| C: Tech + Vol + Sector | 45 | 0.4873 |
| D: Tech + Vol + Sector + Macro | 57 | 0.5274 |
| F: FULL (Inc. KAP/Events) | 62 | 0.5530 |

## 3. Event Impact Analysis (Post-Event Returns)
| Event Type | Samples | Mean Return (5d) | Positive Rate |
|------------|---------|------------------|---------------|
| central_bank | 576 | 1.04% | 52.60% |

## 4. Interaction Analysis
How market regime affects event performance.
| Regime | Has Recent Event | Mean Return | Positive Rate |
|--------|------------------|-------------|---------------|

## 5. Research Conclusions
- **KAP Impact:** Refer to Section 3.
- **Ablation Insight:** Configuration F vs A shows the marginal utility of event data.
- **Limitations:** If events are 0, check `app/data/event_data.py` provider status.
