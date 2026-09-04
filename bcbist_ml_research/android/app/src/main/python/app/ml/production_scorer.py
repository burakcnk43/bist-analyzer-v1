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
        h5_ens = self.models.get(5)
        h5_cal = self.calibrators.get(5)

        regime = market_context.get('regime', 'NORMAL')
        results['regime'] = regime
        trans_prob = market_context.get('transition_prob', 0.0)

        if h5_ens:
            # Phase 11: Base score with Adaptive Weights if provided
            if adaptive_weights:
                h5_ens.set_weights(w_cls=adaptive_weights.get('cls', 0.3),
                                   w_rnk=adaptive_weights.get('rnk', 0.7))

            # 1. Proposal Generation
            specialist_probs = {}
            specialist_probs['general'] = h5_ens.get_alpha_score(X)

            if sector_expert:
                specialist_probs['sector'] = sector_expert.predict(X, h5_ens)
            if regime_expert:
                specialist_probs['regime'] = regime_expert.predict(X, regime, h5_ens)
            if box_expert:
                specialist_probs['box'] = box_expert.predict_probs(X)

            # 2. Gating (Meta V3)
            if meta_v3:
                trust_stats = market_context.get('reliability', {})
                # Box features extraction
                box_cols = [c for c in X.columns if 'box_' in c or 'is_breakout' in c]
                box_metrics = X[box_cols] if box_cols else None

                meta_X = meta_v3.prepare_gating_features(X, specialist_probs, regime, trans_prob, trust_stats, box_metrics)
                results['production_alpha'] = meta_v3.predict_alpha(meta_X)
            else:
                # Fallback to Phase 11 logic
                base_score = specialist_probs['general']
                if meta_v2:
                    scores_dict = {"5d": base_score}
                    if 1 in self.models: scores_dict["1d"] = self.models[1].get_alpha_score(X)
                    if 3 in self.models: scores_dict["3d"] = self.models[3].get_alpha_score(X)

                    meta_X = meta_v2.prepare_meta_features(X, scores_dict, regime, pd.DataFrame())
                    results['production_alpha'] = meta_v2.predict_success_prob(meta_X)
                else:
                    trust = 0.5
                    if success_mdl:
                        X_m = X.copy()
                        X_m['alpha_score'] = base_score
                        X_m['daily_rank'] = pd.Series(base_score, index=X.index).rank(ascending=False)
                        trust = success_mdl.predict_trust(X_m)
                    sim_bin = (np.array(sim_alpha) > 0).astype(float) if sim_alpha is not None else 0.0
                    results['production_alpha'] = (0.5 * base_score) + (0.3 * trust) + (0.2 * sim_bin)

            # 3. Defensive / Recovery Guardrails
            if regime == 'CRASH':
                if 'rel_mkt_return_20' in X.columns:
                    results['production_alpha'] += 0.2 * X['rel_mkt_return_20'].fillna(0)
                if 'rsi_14' in X.columns:
                    results['production_alpha'] *= (X['rsi_14'] / 50.0).clip(0.5, 1.0)

            if regime == 'BULL_OVEREXTENDED' and trans_prob > 0.7:
                # High risk of bubble burst
                results['production_alpha'] *= 0.4
        else:
            results['production_alpha'] = 0

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
