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
    NON_NUMERIC_COLUMNS = ['Seasons', 'Holiday']
    FEATURES_CORRELATION_VALUE = 0.85
    HEATMAP_FOLDER = 'plots'
    HEATMAP_NAME = 'heatmap_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    HEATMAP_PATH = csv_path =\
        Path(__file__).resolve().parents[2] / HEATMAP_FOLDER / HEATMAP_NAME
