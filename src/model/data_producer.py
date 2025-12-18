import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from util.dataframe_utils import DataframeUtils


class DataProducer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> None:
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        consistent_rows = len(self.df)
        if initial_rows > consistent_rows:
            print(f"Cleaned {initial_rows - consistent_rows} rows (duplicates/NaN).")

    def visualize_distribution(self, target: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[target], kde=True, color='blue')
        plt.title(f'Distribution of {target}')
        plt.xlabel('Count')
        plt.ylabel('Frequency')
        plt.savefig(DataframeUtils.DISTRIBUTION_PLOT_PATH)
        plt.show()

    def parse_date(self, date_column: str) -> None:
        self.df[date_column] = pd.to_datetime(self.df[date_column], format='%d/%m/%Y')
        self.df['Month'] = self.df[date_column].dt.month
        self.df['DayOfWeek'] = self.df[date_column].dt.dayofweek
        self.df = self.df.drop(columns=[date_column])

    def drop_unnecessary_data(self, column: str, standing_data_value: str) -> None:
        self.df = self.df[self.df[column] == standing_data_value].copy()
        self.df = self.df.drop(columns=[column])

    def cyclical_encoding(self, column: str) -> None:
        if column in self.df.columns:
            max_val = self.df[column].max() + 1
            self.df[f'{column}_sin'] = (
                np.sin(2 * np.pi * (self.df[column]) / max_val))
            self.df[f'{column}_cos'] = (
                np.cos(2 * np.pi * (self.df[column]) / max_val))
            self.df = self.df.drop(column, axis=1)

    def one_hot_encode(self, columns: list) -> None:
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True, dtype=int)

    def build_heatmap(self) -> None:
        plt.figure(figsize=(16, 12))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(DataframeUtils.HEATMAP_PATH)
        plt.show()

    def drop_highly_correlated_features(self, threshold: float = 0.9) -> None:
        corr_matrix = self.df.corr().abs()

        upper = corr_matrix.where(
                                np.triu(
                                    np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Deleted columns (threshold > {threshold}): {to_drop}")
        self.df = self.df.drop(columns=to_drop)

    def get_feature_names(self, target: str) -> list:
        df = self.df.copy()
        return df.drop(columns=[target]).columns.tolist()
