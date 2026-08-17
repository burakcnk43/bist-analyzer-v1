# KAP Integration Status - BCBIST ML Research Engine

This document outlines the current state of KAP (Public Disclosure Platform) data collection and its impact on the research model.

## 1. Connection & Retrieval Summary
| Metric | Status |
|--------|--------|
| **Primary API Status** | TIMEOUT / UNREACHABLE |
| **RSS Feed Status** | NOT ATTEMPTED (Skipped after API timeout) |
| **CSV Fallback Status** | **ACTIVE & AUTHORITATIVE** |
| **Max Retries** | 2 |
| **Total Response Time** | 64.47s (Graceful failure) |

## 2. Data Coverage (Phase 3)
| Metric | Value |
|--------|-------|
| **Macro Events Collected** | 19 (CBRT Decisions) |
| **Company KAP Events** | 28 (Manual/Real) |
| **Symbols Covered** | 16 / 50 |
| **Event Date Range** | 2023-01-19 to 2024-11-11 |
| **Total Feature Count** | 120 |
| **PIT Validation** | **PASSED** (Strict 18:15 cutoff) |

## 3. Feature Filling Rates
Based on actual dataset audit:
- **Technical/Volume/Sector**: 97% - 100%
- **Macro Indicators**: 66.7% (Due to currency data range)
- **KAP/Event Frequency**: 100.0% (Zero-filled when no events)
- **KAP Recency/Age**: 100.0% (Defaulted to 999.0 for stability)

## 4. Model Impact (Ablation Analysis)
| Configuration | Accuracy (5d) | Incremental Gain |
|---------------|---------------|------------------|
| F: Macro + Tech + Vol | 52.40% | Base |
| G: FULL (Inc. Company KAP) | 52.00% | -0.40% (Noise/Sparsity) |

> [!WARNING]
> **Observation**: Adding company-specific KAP data in its current sparse state (28 events) slightly degraded accuracy (-0.4%). This suggests the model currently treats sparse events as noise. Deeper historical data (1000+ events) is recommended for statistical significance.

## 5. Fallback Instructions
To inject more real data, add rows to `data/raw/kap_events.csv` with the following columns:
- `symbol`: e.g., `THYAO.IS`
- `timestamp`: e.g., `2024-05-15 18:30:00`
- `event_type`: e.g., `dividend`, `contract`, `financial_results`
- `importance`: 1 to 5
- `sentiment`: -1.0 to 1.0 (optional)
