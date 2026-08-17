import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import logging
from typing import Literal

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    Calibrates raw AI scores into real-world probabilities.
    """
    def __init__(self, method: Literal['platt', 'isotonic'] = 'platt'):
        self.method = method
        self.model = None
        if method == 'platt':
            self.model = LogisticRegression(solver='lbfgs')
        else:
            self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        """
        Fits the calibration model using OOF scores and labels.
        """
        scores = scores.reshape(-1, 1) if self.method == 'platt' else scores
        logger.info(f"Fitting {self.method} calibration on {len(scores)} samples...")
        self.model.fit(scores, y_true)

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Applies calibration to new raw scores.
        """
        if self.method == 'platt':
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            return self.model.transform(scores)

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Computes ECE (Expected Calibration Error) and reliability curve data.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # ECE Calculation
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_prop = np.mean(mask)
            ece += bin_prop * np.abs(bin_acc - bin_conf)

    return {
        "ece": ece,
        "reliability_x": prob_pred.tolist(),
        "reliability_y": prob_true.tolist()
    }
