import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class Top5CombinationOptimizer:
    """
    Optimizes the selection of 5 stocks to maximize P(Hits >= 3).
    Second-stage ranker that considers the joint probability and diversification.
    """
    def __init__(self):
        # We'll use a small regressor to predict the expected hit count of a set
        self.model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)

    def select_optimal_set(self, candidates_df: pd.DataFrame, k=5,
                           conditional_scorer=None, market_vector=None,
                           failure_predictor=None, regime='NORMAL',
                           meta_learner=None, weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Maximizes Composite Portfolio Utility using V5 logic.
        Utility = w1*P(3+) + w2*P(4+) + w3*ExpExcessRet - w4*TailRisk - w5*LiquidityPenalty
        """
        if candidates_df.empty: return pd.DataFrame()

        scored = candidates_df.copy()

        # Default V5 Weights (can be learned)
        w = weights or {
            'p3plus': 0.4,
            'p4plus': 0.2,
            'expected_ret': 0.3,
            'tail_risk': 0.2,
            'liquidity': 0.1
        }

        # 1. Base Probabilities & Meta-Adjustments
        if failure_predictor is not None:
            if 'p_fail' in scored.columns:
                # V5 Veto: Stronger threshold for high-risk regimes
                veto_thresh = 0.5 if regime in ['BEAR', 'CRASH'] else 0.65
                scored = scored[scored['p_fail'] < veto_thresh]

        if scored.empty: return pd.DataFrame()

        # 2. Individual Expected Returns & Liquidity
        if 'relative_volume' in scored.columns:
            # V5 Liquidity: factor in spread proxy
            liquidity_score = scored['relative_volume'].clip(0, 1)
            scored['liquidity_penalty'] = (1.0 - liquidity_score) * w['liquidity']
        else:
            scored['liquidity_penalty'] = 0

        scored['joint_prob'] = scored['production_alpha']

        # 3. Dependency-Aware Utility Selection (V5)
        # Ensure no duplicates and unique symbol selection
        scored = scored.sort_values('joint_prob', ascending=False)
        scored = scored.drop_duplicates(subset=['symbol_col'])

        pool = scored.head(25)
        temp_candidates = pool.to_dict('records')

        selected_indices = []
        for _ in range(k):
            best_composite_utility = -100.0
            best_idx = -1

            for i, cand in enumerate(temp_candidates):
                if i in selected_indices: continue

                # Tail Risk Calculation
                tail_risk = 0.0
                if selected_indices:
                    for sel_idx in selected_indices:
                        sel_cand = temp_candidates[sel_idx]
                        # Sector overlap
                        if cand.get('sector') == sel_cand.get('sector'):
                            tail_risk += 0.4 # V5 sector penalty
                        # Macro similarity
                        vol_diff = abs(cand.get('market_z_rolling_std_20', 0) - sel_cand.get('market_z_rolling_std_20', 0))
                        if vol_diff < 0.05:
                            tail_risk += 0.1

                adj_prob = max(0.01, cand['joint_prob'] - (tail_risk * w['tail_risk']))

                # Distributional Probabilities
                current_probs = [temp_candidates[idx].get('effective_prob', 0.5) for idx in selected_indices] + [adj_prob]
                p3plus = self._estimate_p_k_plus(current_probs, target_k=3)
                p4plus = self._estimate_p_k_plus(current_probs, target_k=4)

                # Expected Excess Return proxy
                conviction = cand.get('production_alpha', 0.5)

                # Composite Utility
                utility = (
                    w['p3plus'] * p3plus +
                    w['p4plus'] * p4plus +
                    w['expected_ret'] * conviction -
                    cand['liquidity_penalty']
                )

                if utility > best_composite_utility:
                    best_composite_utility = utility
                    best_idx = i

            if best_idx != -1:
                # Store effective prob for next pick
                penalty = 0
                for sel_idx in selected_indices:
                    if temp_candidates[best_idx].get('sector') == temp_candidates[sel_idx].get('sector'):
                        penalty += 0.4
                temp_candidates[best_idx]['effective_prob'] = max(0.01, temp_candidates[best_idx]['joint_prob'] - penalty)
                selected_indices.append(best_idx)
            else:
                break

        final_picks = [temp_candidates[i] for i in selected_indices]
        return pd.DataFrame(final_picks)

    def _estimate_p_k_plus(self, probs: List[float], target_k=3) -> float:
        """
        Estimates P(Hits >= k) for a set of independent Bernoulli trials.
        Uses recursive DP (exact Poisson-Binomial).
        """
        n = len(probs)
        if n < target_k: return 0.0

        # dp[i][j] is prob of exactly j hits in first i stocks
        dp = np.zeros((n + 1, n + 1))
        dp[0][0] = 1.0

        for i in range(1, n + 1):
            p = probs[i-1]
            for j in range(i + 1):
                # j hits from i can come from:
                # 1. j hits from i-1 and current fails (1-p)
                # 2. j-1 hits from i-1 and current hits (p)
                term1 = dp[i-1][j] * (1 - p) if j <= (i-1) else 0
                term2 = dp[i-1][j-1] * p if j > 0 else 0
                dp[i][j] = term1 + term2

        # Sum all probabilities for j >= target_k
        return float(np.sum(dp[n][target_k:]))

    def estimate_set_hit_prob(self, set_df: pd.DataFrame) -> float:
        """
        Estimates the probability that this set of 5 has >= 3 hits.
        """
        if set_df.empty: return 0.0
        probs = set_df['joint_prob'].values if 'joint_prob' in set_df.columns else set_df['production_alpha'].values
        return self._estimate_p_k_plus(probs, target_k=3)
