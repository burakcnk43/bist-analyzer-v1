import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from app.features.breadth import MarketBreadthEngine
from app.features.sector_rotation import SectorRotationEngine
from app.research.regimes import MarketRegimeDetector, RegimeTransitionModel

logger = logging.getLogger(__name__)

class MarketContextManager:
    """
    Coordinates all market-level intelligence:
    - Breadth
    - Sector Rotation
    - Regime Detection
    - Transition Probabilities
    """
    def __init__(self, models_dir=None):
        self.breadth_engine = MarketBreadthEngine()
        self.sector_engine = SectorRotationEngine()
        self.regime_detector = MarketRegimeDetector()
        self.transition_model = RegimeTransitionModel()
        self.models_dir = models_dir

    def get_daily_context(self, universe_df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculates the complete market context for a given set of stock data.
        Assumes universe_df contains data for multiple stocks on the same date(s).
        """
        if universe_df.empty:
            return {}

        # 1. Breadth
        breadth_stats = self.breadth_engine.calculate_breadth(universe_df)

        # 2. Sector Rotation
        sector_stats = self.sector_engine.calculate_sector_stats(universe_df)

        # 3. Regime
        regime = self.regime_detector.detect_regime(universe_df)

        # 4. Transition Probability
        # Needs historical context to work well, but can use current window
        trans_prob = self.transition_model.predict_transition_prob(universe_df)

        # 5. Aggregate into a context vector
        context = {
            'date': universe_df.index.max(),
            'regime': regime,
            'transition_prob': trans_prob,
            'breadth': breadth_stats.iloc[-1].to_dict() if not breadth_stats.empty else {},
            'sectors': sector_stats[sector_stats['date'] == universe_df.index.max()].to_dict('records')
        }

        return context

    def build_market_feature_matrix(self, global_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a time-series of market-level features for training the DailyHitRatePredictor.
        """
        logger.info("Building Market Feature Matrix...")

        # 1. Breadth Time-series
        breadth_ts = self.breadth_engine.calculate_breadth(global_df)

        # 2. Sector Rotation Time-series (Aggregate the "Is Leader" count and Top sector strength)
        sector_raw = self.sector_engine.calculate_sector_stats(global_df)
        sector_agg = sector_raw.groupby('date').agg(
            max_sector_strength=('sector_relative_strength', 'max'),
            leader_count=('is_leader', 'sum'),
            sector_dispersion=('sector_relative_strength', 'std')
        )

        # 3. Regime and Transitions
        # We'll calculate these row by row for the time series
        regimes = []
        trans_probs = []
        dates = sorted(global_df.index.unique())

        # Optimization: detector is fast, transition model might need a window
        for d in dates:
            day_data = global_df.loc[[d]]
            regimes.append(self.regime_detector.detect_regime(day_data))
            # Transition probability normally uses features available at that date
            # We'll mock or simplify this for the full matrix if transition_model is not trained yet
            trans_probs.append(0.0) # Placeholder, will be populated if model trained

        market_df = pd.DataFrame({
            'regime': regimes,
            'transition_prob': trans_probs
        }, index=dates)

        # Combine
        full_market = market_df.join(breadth_ts).join(sector_agg)

        # Add categorical regime mapping
        regime_map = {"BEAR": -1, "CRASH": -2, "SIDEWAYS": 0, "BULL": 1, "BULL_OVEREXTENDED": 2}
        full_market['regime_val'] = full_market['regime'].map(regime_map).fillna(0)

        return full_market.fillna(0)
