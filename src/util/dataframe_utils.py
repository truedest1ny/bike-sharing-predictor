from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataframeUtils:
    BIKE_DATA_COLUMNS = [
        'Date', 'Rented_Bike_Count', 'Hour', 'Temperature', 'Humidity',
        'Wind_speed', 'Visibility', 'Dew_point_temp', 'Solar_Radiation',
        'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning_Day'
    ]
    TARGET_COLUMN = 'Rented_Bike_Count'
    NON_NUMERIC_COLUMNS = ['Seasons', 'Holiday']
    FEATURES_CORRELATION_VALUE = 0.85
    PLOTS_FOLDER = 'plots'
    PLOTS_NAME_POSTFIX = datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    HEATMAP_NAME = 'heatmap_' + PLOTS_NAME_POSTFIX
    HEATMAP_PATH =\
        Path(__file__).resolve().parents[2] / PLOTS_FOLDER / HEATMAP_NAME
    DISTRIBUTION_PLOT_NAME = 'distribution_' + PLOTS_NAME_POSTFIX
    DISTRIBUTION_PLOT_PATH = \
        Path(__file__).resolve().parents[2] / PLOTS_FOLDER / DISTRIBUTION_PLOT_NAME

