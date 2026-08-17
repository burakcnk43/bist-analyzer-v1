import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict

def analyze_feature_bins(df: pd.DataFrame, feature: str, target: str, bins=10, min_samples=30) -> pd.DataFrame:
    """
    Analyzes the relationship between a feature (binned) and a target with statistical controls.
    """
    if feature not in df.columns or target not in df.columns:
        return pd.DataFrame()

    temp_df = df[[feature, target]].dropna()
    if len(temp_df) < min_samples:
        return pd.DataFrame()

    # Bin the feature
    try:
        temp_df['bin'] = pd.qcut(temp_df[feature], q=bins, duplicates='drop')
    except ValueError:
        temp_df['bin'] = pd.cut(temp_df[feature], bins=bins)

    # Aggregate
    analysis = temp_df.groupby('bin')[target].agg([
        ('sample_count', 'count'),
        ('positive_count', lambda x: (x > 0).sum()),
        ('negative_count', lambda x: (x <= 0).sum()),
        ('mean_forward_return', 'mean'),
        ('median_forward_return', 'median'),
        ('positive_rate', lambda x: (x > 0).mean()),
        ('std', 'std')
    ]).reset_index()

    # Filter by min_samples per bin
    analysis = analysis[analysis['sample_count'] >= max(5, min_samples // bins)]

    # Calculate Confidence Intervals (95%)
    # Mean Return CI
    analysis['se_mean'] = analysis['std'] / np.sqrt(analysis['sample_count'])
    analysis['mean_ci_lower'] = analysis['mean_forward_return'] - 1.96 * analysis['se_mean']
    analysis['mean_ci_upper'] = analysis['mean_forward_return'] + 1.96 * analysis['se_mean']

    # Positive Rate CI (Binomial)
    analysis['pr_se'] = np.sqrt((analysis['positive_rate'] * (1 - analysis['positive_rate'])) / analysis['sample_count'])
    analysis['pr_ci_lower'] = analysis['positive_rate'] - 1.96 * analysis['pr_se']
    analysis['pr_ci_upper'] = analysis['positive_rate'] + 1.96 * analysis['pr_se']

    return analysis

def analyze_stability(splits: List[tuple], feature: str, target: str, bins=5) -> pd.DataFrame:
    """
    Analyzes stability of a feature-target relationship across folds.
    """
    fold_results = []
    for i, (train_df, test_df) in enumerate(splits):
        # We test stability on the TEST set of each fold (out-of-sample stability)
        fold_analysis = analyze_feature_bins(test_df, feature, target, bins=bins, min_samples=20)
        if not fold_analysis.empty:
            fold_analysis['fold'] = i
            fold_results.append(fold_analysis)

    if not fold_results:
        return pd.DataFrame()

    combined = pd.concat(fold_results)

    # Summarize stability per bin
    stability = combined.groupby('bin').agg({
        'mean_forward_return': ['mean', 'std', 'count'],
        'positive_rate': ['mean', 'std']
    }).reset_index()

    stability.columns = ['bin', 'avg_effect', 'effect_std', 'fold_count', 'avg_pos_rate', 'pos_rate_std']
    return stability

def get_top_conditional_patterns(df: pd.DataFrame, features: List[str], target: str, min_samples=30) -> pd.DataFrame:
    results = []
    for feat in features:
        bin_analysis = analyze_feature_bins(df, feat, target, min_samples=min_samples)
        if not bin_analysis.empty:
            bin_analysis['feature'] = feat
            results.append(bin_analysis)

    if not results:
        return pd.DataFrame()

    return pd.concat(results).sort_values(by='mean_forward_return', ascending=False)
