import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import joblib

from app.config import MODELS_DIR, DATA_REPORTS_DIR
from app.ml.production_scorer import ProductionScorer
from app.ml.selection import Top5CombinationOptimizer
from app.ml.adaptive_meta import AdaptiveMetaLearnerV5
from app.ml.transition_expert import RegimeTransitionExpert
from app.ml.monitoring import DriftGuardV2
from app.ml.market_quality import HighQualityMarketDayPredictor
from app.ml.adaptive_k import AdaptiveTopKSelectorV2
from app.research.regimes import MarketRegimeDetector

logger = logging.getLogger(__name__)

class ProductionManager:
    """
    Champion Decision Engine (Phase 22).
    Orchestrates data ingestion, multi-expert scoring, risk vetting,
    and utility-optimized selection.
    """
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.scorer = ProductionScorer(MODELS_DIR)
        self.optimizer = Top5CombinationOptimizer()
        self.meta_learner = AdaptiveMetaLearnerV5(
            windows=self.config['meta_learning']['windows'],
            decay_factor=self.config['meta_learning']['decay_factor']
        )
        self.transition_expert = RegimeTransitionExpert()
        self.drift_guard = DriftGuardV2(threshold=self.config['risk']['max_drift_psi'])
        self.quality_predictor = HighQualityMarketDayPredictor()
        self.k_selector = AdaptiveTopKSelectorV2()
        self.regime_detector = MarketRegimeDetector()

    def get_daily_picks(self,
                        day_data: pd.DataFrame,
                        market_data: pd.DataFrame,
                        historical_trust_data: Optional[Dict] = None) -> Dict:
        """
        Main entry point for production predictions.
        """
        current_date = day_data.index.max()
        regime = self.regime_detector.detect_regime(market_data)

        # 1. Intelligence Layer
        trans_prob = self.transition_expert.predict_transition_prob(market_data)
        mkt_quality = self.quality_predictor.calculate_quality_score(market_data.iloc[-1], regime)
        drift_score = self.drift_guard.get_global_drift_score(day_data)

        # 2. Adaptive Weighting
        rel_feats = self.meta_learner.get_contextual_reliability(current_date, regime)
        # Apply reliability adjustment to base scores
        trust_coeff = rel_feats.get('rel_general_20d', 0.5) / 0.5

        # 3. Expert Scoring
        scores_df = self.scorer.calculate_production_scores(day_data, {'regime': regime})
        day_scored = day_data.join(scores_df)
        day_scored['production_alpha'] = (day_scored['production_alpha'] * trust_coeff).clip(0, 1)

        # 4. Utility-Based K Selection
        # (Using a simplified hit distribution for the live selector)
        hit_dist = {1: 0.8, 3: 0.4, 5: 0.2} # This should ideally come from GroupOutcomePredictorV2
        k = self.k_selector.determine_optimal_k(hit_dist, mkt_quality)

        if k == 0:
            return self._format_abstain_response(current_date, regime, mkt_quality, drift_score)

        # 5. V5 Portfolio Optimization
        picks = self.optimizer.select_optimal_set(
            day_scored, k=k,
            regime=regime,
            weights=self.config['selection']['utility_weights']
        )

        return self._format_success_response(current_date, regime, mkt_quality, drift_score, picks, k)

    def _format_abstain_response(self, date, regime, quality, drift):
        return {
            "date": date.isoformat(),
            "market_regime": regime,
            "market_quality": quality,
            "drift_score": drift,
            "selected_k": 0,
            "predictions": [],
            "status": "ABSTAIN",
            "reason": "Market risk exceeded defensive thresholds."
        }

    def _format_success_response(self, date, regime, quality, drift, picks, k):
        predictions = []
        for i, (_, row) in enumerate(picks.iterrows()):
            predictions.append({
                "symbol": row['symbol_col'],
                "rank": i + 1,
                "probability": float(row['production_alpha']),
                "sector": row.get('sector', 'Other'),
                "utility": float(row.get('joint_prob', 0.5)),
                "reason": self._generate_deterministic_reason(row)
            })

        return {
            "date": date.isoformat(),
            "market_regime": regime,
            "market_quality": quality,
            "drift_score": drift,
            "selected_k": k,
            "predictions": predictions,
            "status": "SUCCESS"
        }

    def _generate_deterministic_reason(self, row: pd.Series) -> str:
        reasons = []
        if row.get('relative_volume', 0) > 1.2: reasons.append("Strong relative volume confirmation")
        if row.get('dist_sma_20', 0) > 0: reasons.append("Trading above 20-day mean")
        if row.get('box_staircase_score', 0) > 2: reasons.append("Darvas staircase maturity")
        if row.get('production_alpha', 0) > 0.8: reasons.append("High multi-expert consensus")

        return " | ".join(reasons) if reasons else "Selected by composite utility optimization."
