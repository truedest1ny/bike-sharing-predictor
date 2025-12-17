from typing import Any

import numpy as np
import pandas as pd


class DataNormalizer:
    def __init__(self, scaler: Any):
        self.scaler = scaler

    def fit_normalize_train(self, x_train: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(x_train)

    def normalize_test(self, x_test: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(x_test)
