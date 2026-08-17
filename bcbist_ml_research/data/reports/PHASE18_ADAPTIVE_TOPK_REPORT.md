# PHASE 18: ADAPTIVE TOP-K PERFORMANCE

## Comparison Table
| Strategy | 3PLUS (Uncond) | Avg K | Coverage |
| :--- | :--- | :--- | :--- |
| Forced Top-5 | 90.41% | 5.0 | 100% |
| Adaptive Top-K | 20.55% | 1.33 | 56% |

> [!WARNING]
> Selecting K < 3 mathematically prevents 3PLUS success. The Adaptive strategy's 3PLUS score is artificially low because it optimized for individual precision (K=1) on many days.

## Recommendation
For production, use **Adaptive Top-K** with a constraint that $K \in \{0, 3, 5\}$.
