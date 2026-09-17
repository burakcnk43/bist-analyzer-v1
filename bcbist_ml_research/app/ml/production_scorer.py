import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ProductionScorer:
    """
    Production selection layer.
    Features:
    - Multi-Horizon Calibration
    - Meta-Ensemble (Base + SuccessTrust + Similarity)
    - Defensive Regime Guard (Abstain mechanism)
    - Sector Diversification
    """
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}
        self.calibrators = {}
        self.horizons = [1, 3, 5]
        self._load_assets()

    def _load_assets(self):
        from app.ml.ensemble import AlphaEnsemble
        for h in self.horizons:
            cls_path = self.models_dir / f"phase10_cls_{h}d.joblib"
            rnk_path = self.models_dir / f"phase10_rnk_{h}d.joblib"
            if cls_path.exists() and rnk_path.exists():
                p_cls = joblib.load(cls_path)
                p_rnk = joblib.load(rnk_path)
                self.models[h] = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

            cal_path = self.models_dir / f"phase10_calibrator_{h}d.joblib"
            if cal_path.exists():
                self.calibrators[h] = joblib.load(cal_path)

    def calculate_production_scores(self, X: pd.DataFrame,
                                   market_context: Dict,
                                   success_mdl=None,
                                   sim_alpha=None,
                                   meta_v2=None,
                                   adaptive_weights: Optional[Dict] = None,
                                   sector_expert=None,
                                   regime_expert=None,
                                   box_expert=None,
                                   meta_v3=None) -> pd.DataFrame:
        results = pd.DataFrame(index=X.index)

        # Phase 26 Pivot: Prioritize 1D and 3D horizons for maximum immediate accuracy
        h1_ens = self.models.get(1)
        h3_ens = self.models.get(3)
        h1_cal = self.calibrators.get(1)

        regime = market_context.get('regime', 'NORMAL')
        results['regime'] = regime
        trans_prob = market_context.get('transition_prob', 0.0)

        if h1_ens and h3_ens:
            # 1. Proposal Generation (Precision Momentum focus)
            specialist_probs = {}
            # Blend 1D and 3D for the 'general' expert to capture ultra-short pulse
            s1 = h1_ens.get_alpha_score(X)
            s3 = h3_ens.get_alpha_score(X)
            specialist_probs['general'] = (0.7 * s1 + 0.3 * s3)

            if sector_expert:
                specialist_probs['sector'] = sector_expert.predict(X, h1_ens)
            if regime_expert:
                specialist_probs['regime'] = regime_expert.predict(X, regime, h1_ens)
            if box_expert:
                specialist_probs['box'] = box_expert.predict_probs(X)

            # 2. Gating (Meta V3 - Targeted for 1D)
            if meta_v3:
                trust_stats = market_context.get('reliability', {})
                box_cols = [c for c in X.columns if 'box_' in c or 'is_breakout' in c]
                box_metrics = X[box_cols] if box_cols else None

                meta_X = meta_v3.prepare_gating_features(X, specialist_probs, regime, trans_prob, trust_stats, box_metrics)
                results['production_alpha'] = meta_v3.predict_alpha(meta_X)
            else:
                results['production_alpha'] = specialist_probs['general']

            # 3. Defensive / Ultra-Short Guardrails
            # If 1D trend is negative, penalize alpha heavily for Phase 26
            if 'return_1d' in X.columns:
                results['production_alpha'] *= np.where(X['return_1d'] < 0, 0.8, 1.0)

            # Calibrate to 1D probability if available
            if h1_cal:
                results['calibrated_prob'] = h1_cal.calibrate(results['production_alpha'].values)
            else:
                results['calibrated_prob'] = results['production_alpha']
        else:
            results['production_alpha'] = 0
            results['calibrated_prob'] = 0

        return results

    def select_top_5(self, scored_df: pd.DataFrame, meta_info: pd.DataFrame,
                     optimizer=None) -> pd.DataFrame:
        df = scored_df.join(meta_info)
        if df.empty: return pd.DataFrame()

        regime = df['regime'].iloc[0]

        # 1. Selection Mechanism
        if optimizer:
            return optimizer.select_optimal_set(df, k=5)

        # Fallback to standard diversified selection
        # Phase 12: Dynamic Diversification
        if regime in ['BEAR', 'CRASH']:
            max_per_sector = 1
        elif regime == 'SIDEWAYS':
            max_per_sector = 2
        else: # BULL
            max_per_sector = 3

        # Phase 11 Logic: Abstain only in extreme crash or bubble
        if regime == 'BULL_OVEREXTENDED' and (df['production_alpha'] < 0.7).all():
            logger.warning("Bubble detected. No high-conf picks. ABSTAIN.")
            return pd.DataFrame()

        if regime == 'CRASH' and (df['production_alpha'] < 0.6).all():
             logger.warning("Market crash. No defensive winners found. ABSTAIN.")
             return pd.DataFrame()

        # Standard Selection
        picks = []
        candidates = df.sort_values('production_alpha', ascending=False).copy()

        sector_counts = {}
        for _, row in candidates.iterrows():
            if len(picks) >= 5: break
            sector = row.get('sector_col', 'Other')
            if sector_counts.get(sector, 0) < max_per_sector:
                picks.append(row)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return pd.DataFrame(picks)
