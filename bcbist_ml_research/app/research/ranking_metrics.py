import pandas as pd
import numpy as np

def calculate_top_k_metrics(df: pd.DataFrame, score_col: str, return_col: str, k=5):
    """
    Calculates hit rate and mean return for the top K ranked items per day.
    """
    def get_group_stats(group_df):
        top_k = group_df.nlargest(k, score_col)

        # Hit is defined as positive return
        hit_count = (top_k[return_col] > 0).sum()
        avg_ret = top_k[return_col].mean()

        # Benchmark: All stocks on that day
        mkt_avg = group_df[return_col].mean()

        return pd.Series({
            'hit_rate': hit_count / k,
            'mean_return': avg_ret,
            'excess_return': avg_ret - mkt_avg,
            'hits': int(hit_count)
        })

    daily_stats = df.groupby(level=0).apply(get_group_stats)

    # Hit Distribution: % of days where hits == X
    hit_dist = daily_stats['hits'].value_counts(normalize=True).sort_index().to_dict()
    # Ensure all possible values 0..k are in dict
    full_dist = {i: hit_dist.get(float(i), 0.0) for i in range(k + 1)}

    summary = {
        'avg_hit_rate': daily_stats['hit_rate'].mean(),
        'avg_return': daily_stats['mean_return'].mean(),
        'avg_excess_return': daily_stats['excess_return'].mean(),
        'percent_days_positive': (daily_stats['mean_return'] > 0).mean(),
        'hit_distribution': full_dist
    }

    return summary, daily_stats
