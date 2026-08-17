import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ResearchTrialRegistry:
    """
    Logs every experiment configuration and result to prevent multiple testing bias.
    Calculates a 'Research Significance Discount' for findings.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.trials = []
        self._load_existing()

    def _load_assets(self):
         if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                self.trials = json.load(f)

    def _load_existing(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    self.trials = json.load(f)
            except:
                self.trials = []

    def record_trial(self,
                     experiment_id: str,
                     config: Dict,
                     metrics: Dict,
                     accepted: bool,
                     rejection_reason: str = ""):
        """
        Records a single trial with metadata.
        """
        trial = {
            'timestamp': datetime.now().isoformat(),
            'trial_id': experiment_id,
            'config': config,
            'metrics': metrics,
            'accepted': accepted,
            'reason': rejection_reason
        }
        self.trials.append(trial)
        self._save()

        logger.info(f"Recorded trial: {experiment_id} (Accepted: {accepted})")

    def _save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.trials, f, indent=4)

    def calculate_pbo_discount(self) -> float:
        """
        Calculates a discount factor for the reported performance
        based on the number of trials and effective degrees of freedom.
        """
        n_trials = len(self.trials)
        if n_trials < 5: return 1.0

        # Simple Bonferroni-inspired discount for research bias
        # discount = 1 / log10(n_trials + 1)
        discount = 1.0 / np.log10(n_trials + 9)
        return float(np.clip(discount, 0.5, 1.0))

    def generate_registry_report(self) -> pd.DataFrame:
        """
        Returns all trials as a DataFrame.
        """
        return pd.DataFrame(self.trials)
