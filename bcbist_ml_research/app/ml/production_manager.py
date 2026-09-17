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
from app.ml.group_failure import GroupOutcomePredictorV2
from app.ml.meta_labeler import AlphaTrustModel
from app.ml.box_conditional import DarvasQualityScorer
from app.ml.event_chains import EventChainExpert
from app.ml.monitoring import DriftGuardV2
from app.ml.market_quality import HighQualityMarketDayPredictor
from app.ml.adaptive_k import AdaptiveTopKSelectorV2
from app.research.regimes import MarketRegimeDetector

from app.ml.intraday_filter import IntradayGater
from app.ml.pdf_reports import BcbistPdfEngine

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
        prod_models_dir = MODELS_DIR / "production"

        self.meta_learner = AdaptiveMetaLearnerV5(
            windows=self.config['meta_learning']['windows'],
            decay_factor=self.config['meta_learning']['decay_factor']
        )
        self.meta_learner.load_state(prod_models_dir / "meta_learner_state.joblib")

        # Load Experts
        self.transition_expert = RegimeTransitionExpert.load(prod_models_dir / "transition_expert.joblib") if (prod_models_dir / "transition_expert.joblib").exists() else RegimeTransitionExpert()
        self.group_predictor = GroupOutcomePredictorV2.load(prod_models_dir / "group_outcome_v2.joblib") if (prod_models_dir / "group_outcome_v2.joblib").exists() else GroupOutcomePredictorV2()
        self.trust_model = AlphaTrustModel.load(prod_models_dir / "alpha_trust.joblib") if (prod_models_dir / "alpha_trust.joblib").exists() else AlphaTrustModel()
        self.darvas_scorer = DarvasQualityScorer.load(prod_models_dir / "darvas_quality.joblib") if (prod_models_dir / "darvas_quality.joblib").exists() else DarvasQualityScorer()
        self.event_expert = EventChainExpert.load(prod_models_dir / "event_chain.joblib") if (prod_models_dir / "event_chain.joblib").exists() else EventChainExpert()

        self.drift_guard = DriftGuardV2(threshold=self.config['risk']['max_drift_psi'])
        self.quality_predictor = HighQualityMarketDayPredictor()
        self.k_selector = AdaptiveTopKSelectorV2(
            min_utility_threshold=self.config['selection'].get('min_utility_threshold', 0.15)
        )
        self.regime_detector = MarketRegimeDetector()
        self.intraday_gater = IntradayGater()
        self.pdf_engine = BcbistPdfEngine(DATA_REPORTS_DIR / "daily")

    def get_daily_picks(self,
                        day_data: pd.DataFrame,
                        market_data: pd.DataFrame,
                        intraday_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """
        Main entry point for production predictions.
        Includes Intraday Confirmation Step.
        """
        current_date = day_data.index.max()
        regime = self.regime_detector.detect_regime(market_data)

        # 1. Intelligence Layer
        trans_prob = self.transition_expert.predict_transition_prob(market_data)
        mkt_quality = self.quality_predictor.calculate_quality_score(market_data.iloc[-1], regime)
        drift_score = self.drift_guard.get_global_drift_score(day_data)

        # 2. Expert Scoring & Meta-Adjustments
        scores_df = self.scorer.calculate_production_scores(day_data, {'regime': regime})
        day_scored = day_data.join(scores_df).fillna(0)

        # Specialist Multipliers
        p_trust = self.trust_model.predict_trust_score(day_scored)
        p_darvas = self.darvas_scorer.predict_success_prob(day_scored)
        p_event = self.event_expert.predict_event_impulse(day_scored)

        # Weighted Ensemble for Production Alpha (Phase 26 Ultra-Short Pivot)
        # Prioritizes 1-2 day momentum and reliability
        day_scored['production_alpha'] = (
            0.40 * day_scored['production_alpha'] + # 1D/3D Base Blend
            0.30 * p_trust +                        # AlphaTrust Precision
            0.15 * p_darvas +                       # Darvas Box Breakout
            0.15 * p_event                          # Causal Sentiment Impulse
        ).clip(0, 1)

        # Adaptive Reliability adjustment
        rel_feats = self.meta_learner.get_contextual_reliability(current_date, regime)
        trust_coeff = rel_feats.get('rel_general_20d', 0.5) / 0.5
        day_scored['production_alpha'] = (day_scored['production_alpha'] * trust_coeff).clip(0, 1)

        # 3. Intraday Confirmation Veto (Phase 23)
        if intraday_data:
            logger.info("Applying Intraday Confirmation Veto...")
            day_scored = self.intraday_gater.filter_candidates(day_scored, intraday_data)

        # 4. Utility-Based K Selection
        group_X = self.group_predictor.prepare_group_features(market_data.iloc[-1], day_scored)
        hit_dist = self.group_predictor.predict_hit_distribution(group_X)
        k = self.k_selector.determine_optimal_k(hit_dist, mkt_quality)

        if k == 0 or day_scored.empty:
            return self._format_abstain_response(current_date, regime, mkt_quality, drift_score)

        # 5. V5 Portfolio Optimization
        picks = self.optimizer.select_optimal_set(
            day_scored, k=k,
            regime=regime,
            weights=self.config['selection']['utility_weights']
        )

        response = self._format_success_response(current_date, regime, mkt_quality, drift_score, picks, k)

        # Auto-generate PDF report
        self.pdf_engine.generate_daily_pdf(response)

        return response

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
        if row.get('rel_sector_return_1d', 0) > 0.01: reasons.append("Sector relative strength leader")

        return " | ".join(reasons) if reasons else "Selected by composite utility optimization."
