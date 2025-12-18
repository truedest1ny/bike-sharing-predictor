import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from util.visualizer_utils import VisualizerUtils


class Visualizer:
    @staticmethod
    def visualize_comparing(df: pd.DataFrame):
        models = df['Model'].tolist()
        metrics_to_plot = [col for col in df.columns if col != 'Model']

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            x_pos = np.arange(len(models))
            bars = ax.bar(models, df[metric],
                          color=VisualizerUtils.ML_PLOT_BAR_COLORS[idx],
                          alpha=0.8, edgecolor='black')

            ax.text(0.5, 0.97, f'{metric} Comparison',
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    ha='center', va='top',
                    )

            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=12, ha='right', fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)

            max_val = df[metric].max()
            for bar in bars:
                height = bar.get_height()
                fmt_str = f'{height:.2f}' if metric != 'R2' else f'{height:.4f}'
                ax.annotate(fmt_str,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

            ax.set_ylim(
                df[metric].min() * 0.8 if df[metric].min() < 0 else 0, max_val * 1.15)

        fig.suptitle('Regression Models Performance Comparison',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(VisualizerUtils.ML_PLOT_PATH)
        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names: list):

        if not hasattr(model, 'feature_importances_'):
            print(f"Model {type(model).__name__} does not support feature_importances_")
            return

        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        y_labels = [feature_names[i] for i in indices]
        x_values = importance[indices]

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x=x_values,
            y=y_labels,
            hue=y_labels,
            palette='viridis',
            legend=False
        )

        plt.title(
            'Top Features Impacting Bike Rental Demand', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(VisualizerUtils.FEATURES_IMPORTANCE_PLOT_PATH)
        plt.show()

    @staticmethod
    def plot_regression_results(y_test, y_predicted, model_name: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_predicted, alpha=0.3)

        min_val = min(y_test.min(), y_predicted.min())
        max_val = max(y_test.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2)

        plt.title(f'Actual vs Predicted - {model_name}')
        plt.xlabel('Actual Count')
        plt.ylabel('Predicted Count')
        plt.savefig(VisualizerUtils.RESULTS_PLOT_PATH)
        plt.show()
