# Phase 8: Success Pattern Mining Report

## 1. Discovered Decision Rules (Top-10 Picks)
These rules describe conditions where Top-10 model picks were more likely to be Hits (>0% return).
```
|--- market_z_rsi_14 <= 3.13
|   |--- macro_usdtry_close <= 46.46
|   |   |--- return_3d <= 0.05
|   |   |   |--- class: 1
|   |   |--- return_3d >  0.05
|   |   |   |--- class: 1
|   |--- macro_usdtry_close >  46.46
|   |   |--- rsi_14_slope_3 <= 1.51
|   |   |   |--- class: 0
|   |   |--- rsi_14_slope_3 >  1.51
|   |   |   |--- class: 0
|--- market_z_rsi_14 >  3.13
|   |--- ma_slope_200 <= 0.04
|   |   |--- rsi_7 <= 89.55
|   |   |   |--- class: 1
|   |   |--- rsi_7 >  89.55
|   |   |   |--- class: 1
|   |--- ma_slope_200 >  0.04
|   |   |--- dmp_14 <= 49.79
|   |   |   |--- class: 1
|   |   |--- dmp_14 >  49.79
|   |   |   |--- class: 1

```

## 2. Sector-Specific Hit Rates (Top-5 Only)
| sector_col   |   hit_rate |   sample_count |   avg_return |
|:-------------|-----------:|---------------:|-------------:|
| Other        |   0.611594 |           2070 |    0.0327728 |

## 3. Indicator Optimization (Quantile Analysis)
### RSI 14 Performance Buckets
| bucket                 |   is_hit |   target_return_5d |   symbol_col |
|:-----------------------|---------:|-------------------:|-------------:|
| (-0.000999932, 73.518] | 0.495169 |         0.00971727 |          414 |
| (73.518, 78.101]       | 0.586957 |         0.0239515  |          414 |
| (78.101, 82.344]       | 0.625604 |         0.0412523  |          414 |
| (82.344, 87.12]        | 0.635266 |         0.0323149  |          414 |
| (87.12, 100.0]         | 0.714976 |         0.0566283  |          414 |

### Relative Volume Performance Buckets
| bucket          |   is_hit |   target_return_5d |   symbol_col |
|:----------------|---------:|-------------------:|-------------:|
| (-0.001, 0.394] | 0.591787 |          0.0599906 |          414 |
| (0.394, 0.598]  | 0.625604 |          0.0342243 |          414 |
| (0.598, 0.788]  | 0.620773 |          0.0306026 |          414 |
| (0.788, 1.077]  | 0.620773 |          0.0198205 |          414 |
| (1.077, 20.0]   | 0.599034 |          0.0192262 |          414 |

