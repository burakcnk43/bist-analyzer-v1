import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AdaptiveTopKSelectorV2:
    """
    Dynamically determines K in {1, 3, 5} to maximize Expected Group Utility.
    """
    def __init__(self, min_utility_threshold=0.3):
        self.min_utility_threshold = min_utility_threshold

    def determine_optimal_k(self,
                             hit_distribution: Dict[int, float],
                             market_quality: float) -> int:
        """
        Calculates Utility for K=1, 3, 5.
        Utility = P(Success at K) * (Expected return coefficient)
        """
        # Hit probabilities
        p1plus = hit_distribution.get(1, 0.5)
        p2_of_3 = hit_distribution.get(2, 0.3) # P(Hits >= 2) for K=3 approximation
        p3_of_5 = hit_distribution.get(3, 0.1)

        # Utilities (Higher weights for group targets)
        u1 = p1plus * 0.5
        u3 = p2_of_3 * 0.9
        u5 = p3_of_5 * 1.5

        # Add Market Quality factor
        mq_factor = market_quality / 100.0
        u3 *= (0.3 + 0.7 * mq_factor)
        u5 *= mq_factor

        utilities = {1: u1, 3: u3, 5: u5}
        best_k = max(utilities, key=utilities.get)

        # Abstention check if best utility is too low
        if utilities[best_k] < self.min_utility_threshold:
            return 0

        return best_k
