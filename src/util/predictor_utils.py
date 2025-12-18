from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class PredictorUtils:
    TARGET_COLUMN = 'Rented_Bike_Count'
    TEST_SIZE = 0.1
    RANDOM_SEED = 42
    MODEL_N_ESTIMATORS = 4000
    MODEL_N_NEIGHBORS = 7

    MODELS = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=MODEL_N_ESTIMATORS,
                                               random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=MODEL_N_ESTIMATORS,
                                                       random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=MODEL_N_NEIGHBORS)
    }

    METRICS = {
            'Model': [],
            'RMSE': [],
            'MAE': [],
            'R2': [],
            'MSE': []
    }
