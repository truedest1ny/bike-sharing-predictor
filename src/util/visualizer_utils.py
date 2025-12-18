from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class VisualizerUtils:
    ML_PLOT_BAR_COLORS = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    PLOT_FOLDER = 'plots'
    PLOT_NAME_POSTFIX = datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    ML_PLOT_NAME = 'ml_plot_' + PLOT_NAME_POSTFIX
    ML_PLOT_PATH = \
        Path(__file__).resolve().parents[2] / PLOT_FOLDER / ML_PLOT_NAME
    FEATURES_IMPORTANCE_PLOT_NAME =\
        'features_plot_' + PLOT_NAME_POSTFIX
    FEATURES_IMPORTANCE_PLOT_PATH =\
        Path(__file__).resolve().parents[2] / PLOT_FOLDER / FEATURES_IMPORTANCE_PLOT_NAME
    RESULTS_PLOT_NAME = 'results_plot_' + PLOT_NAME_POSTFIX
    RESULTS_PLOT_PATH = Path(__file__).resolve().parents[2] / PLOT_FOLDER / RESULTS_PLOT_NAME
