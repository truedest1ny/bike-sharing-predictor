from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class PredictorUtils:
    """
        Configuration utility for machine learning models and evaluation metrics.

        This class centralizes model hyperparameters, defines the suite of
        regression algorithms to be tested, and initializes the structure
        for performance tracking.
    """
    # Target variable to be predicted by the regression models
    TARGET_COLUMN = 'Rented_Bike_Count'

    # Proportion of the dataset reserved for testing (now 10%)
    TEST_SIZE = 0.1

    # Seed for random number generators to ensure reproducible results
    RANDOM_SEED = 42

    # Hyperparameter: Number of trees in ensemble models (Forest/Boosting)
    MODEL_N_ESTIMATORS = 4000

    # Hyperparameter: Number of nearest neighbors for the k-NN algorithm
    MODEL_N_NEIGHBORS = 7

    # Dictionary of initialized scikit-learn regressor objects.
    # Includes linear, tree-based, and instance-based algorithms for comparison.
    MODELS = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=MODEL_N_ESTIMATORS,
                                               random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=MODEL_N_ESTIMATORS,
                                                       random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=MODEL_N_NEIGHBORS)
    }

    # Template for storing performance scores during the evaluation phase
    METRICS = {
            'Model': [],
            'RMSE': [],
            'MAE': [],
            'R2': [],
            'MSE': []
    }
