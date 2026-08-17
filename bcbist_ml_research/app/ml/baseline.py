import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

class BaselineModel:
    """
    Simple baseline: predict based on last known direction or mean.
    """
    def predict_direction(self, df: pd.DataFrame) -> np.ndarray:
        # Simplest: predict next day same as last day
        return (df['return_1d'] > 0).astype(int).values

    def predict_return(self, df: pd.DataFrame) -> np.ndarray:
        # Simplest: predict 0 return
        return np.zeros(len(df))
