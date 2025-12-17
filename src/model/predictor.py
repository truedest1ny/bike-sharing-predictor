import copy

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import pandas as pd

from model.data_normalizer import DataNormalizer
from util.predictor_utils import PredictorUtils


class Predictor:
    def __init__(self, df: pd.DataFrame, normalizer: DataNormalizer, models: dict):
        self.df = df
        self.normalizer = normalizer
        self.models = models

    def predict(self) -> pd.DataFrame:
        x_train, x_test, y_train, y_test = self._prepare(
            PredictorUtils.TARGET_COLUMN,
            PredictorUtils.TEST_SIZE,
            PredictorUtils.RANDOM_SEED)

        metrics = copy.deepcopy(PredictorUtils.METRICS)

        for model_name, model in self.models.items():
            y_predicted = model.fit(x_train, y_train).predict(x_test)

            mse = mean_squared_error(y_test, y_predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_predicted)
            r2 = r2_score(y_test, y_predicted)

            metrics['Model'].append(model_name)
            metrics['MSE'].append(mse)
            metrics['RMSE'].append(rmse)
            metrics['MAE'].append(mae)
            metrics['R2'].append(r2)

        return pd.DataFrame(metrics).sort_values(by='R2', ascending=False)


    def _prepare(self, target: str,
                 test_size: float,
                 random_state: int) -> tuple:
        x = self.df.drop(columns=[target])
        y = self.df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        x_train_scaled = self.normalizer.fit_normalize_train(x_train)
        x_test_scaled = self.normalizer.normalize_test(x_test)

        return x_train_scaled, x_test_scaled, y_train, y_test
