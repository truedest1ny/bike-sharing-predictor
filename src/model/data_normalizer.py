from typing import Any

import numpy as np
import pandas as pd


class DataNormalizer:
    """
        Handles feature scaling and normalization.

        This class wraps scikit-learn scalers to ensure consistent data
        transformation across training and testing sets, preventing
        information leakage by fitting only on the training data.
    """
    def __init__(self, scaler: Any):
        """
            Initializes the DataNormalizer with a specific scaling strategy.

            Args:
                scaler (Any): A scikit-learn-compatible scaler object
                                (e.g., StandardScaler, MinMaxScaler).
        """
        self.scaler = scaler

    def fit_normalize_train(self, x_train: pd.DataFrame) -> np.ndarray:
        """
            Fits the scaler to the training data and returns the transformed features.

            This method calculates the necessary parameters (like mean and variance)
            from the training set and applies the transformation.

            Args:
                x_train (pd.DataFrame): The training feature matrix.

            Returns:
                np.ndarray: The scaled training data.
        """
        return self.scaler.fit_transform(x_train)

    def normalize_test(self, x_test: pd.DataFrame) -> np.ndarray:
        """
            Transforms the test data using parameters learned from the training set.

            It is crucial NOT to refit the scaler on test data to maintain
            the integrity of the evaluation and avoid data leakage.

            Args:
                x_test (pd.DataFrame): The testing feature matrix.

            Returns:
                np.ndarray: The scaled testing data.
        """
        return self.scaler.transform(x_test)
