from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataframeUtils:
    """
        Centralized configuration for dataset structure and storage paths.

        This dataclass acts as a single source of truth for column names,
        preprocessing thresholds, and dynamic file paths for generated plots.
    """

    # Standardized column names for the Seoul Bike Sharing Demand dataset
    BIKE_DATA_COLUMNS = [
        'Date', 'Rented_Bike_Count', 'Hour', 'Temperature', 'Humidity',
        'Wind_speed', 'Visibility', 'Dew_point_temp', 'Solar_Radiation',
        'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning_Day'
    ]

    # The dependent variable for regression analysis
    TARGET_COLUMN = 'Rented_Bike_Count'

    # Categorical features requiring encoding before model training
    NON_NUMERIC_COLUMNS = ['Seasons', 'Holiday']

    # Threshold for dropping features with high multicollinearity
    FEATURES_CORRELATION_VALUE = 0.85

    # Directory settings for output visualizations
    PLOTS_FOLDER = 'plots'

    # Timestamped postfix to prevent overwriting results from different runs
    PLOTS_NAME_POSTFIX = datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'

    # Dynamic path resolution for the correlation heatmap
    HEATMAP_NAME = 'heatmap_' + PLOTS_NAME_POSTFIX
    HEATMAP_PATH =\
        Path(__file__).resolve().parents[2] / PLOTS_FOLDER / HEATMAP_NAME

    # Dynamic path resolution for the target variable distribution plot
    DISTRIBUTION_PLOT_NAME = 'distribution_' + PLOTS_NAME_POSTFIX
    DISTRIBUTION_PLOT_PATH = \
        Path(__file__).resolve().parents[2] / PLOTS_FOLDER / DISTRIBUTION_PLOT_NAME
