from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class VisualizerUtils:
    """
        Configuration utility for plot styling and storage paths.

        This class manages the aesthetic properties of visualizations and
        resolves dynamic file paths to ensure all generated plots are
        organized and timestamped correctly.
    """

    # Specific color palette for the 4-panel metrics comparison bar chart
    ML_PLOT_BAR_COLORS = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

    # Directory name where all visualization artifacts will be stored
    PLOT_FOLDER = 'plots'

    # Timestamped postfix to ensure unique filenames for every execution run
    PLOT_NAME_POSTFIX = datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'

    # Dynamic path resolution for the Model Comparison plot (Metrics)
    ML_PLOT_NAME = 'ml_plot_' + PLOT_NAME_POSTFIX
    ML_PLOT_PATH = \
        Path(__file__).resolve().parents[2] / PLOT_FOLDER / ML_PLOT_NAME

    # Dynamic path resolution for the Feature Importance plot
    FEATURES_IMPORTANCE_PLOT_NAME =\
        'features_plot_' + PLOT_NAME_POSTFIX
    FEATURES_IMPORTANCE_PLOT_PATH =\
        Path(__file__).resolve().parents[2] / PLOT_FOLDER / FEATURES_IMPORTANCE_PLOT_NAME

    # Dynamic path resolution for the Actual vs Predicted regression results plot
    RESULTS_PLOT_NAME = 'results_plot_' + PLOT_NAME_POSTFIX
    RESULTS_PLOT_PATH =\
        Path(__file__).resolve().parents[2] / PLOT_FOLDER / RESULTS_PLOT_NAME
