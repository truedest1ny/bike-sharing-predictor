import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from util.dataframe_utils import DataframeUtils


class DataProducer:
    """
        Handles comprehensive data preprocessing, feature engineering, and EDA.

        This class transforms raw data into a format suitable for machine learning,
        managing cleaning, categorical encoding, cyclical transformations,
        and feature selection based on correlation.
    """
    def __init__(self, df: pd.DataFrame):
        """
            Initializes the DataProducer with a DataFrame.

            Args:
                df (pd.DataFrame): The raw input dataset.
        """
        self.df = df

    def clean_data(self) -> None:
        """
            Removes duplicates and handles missing values (NaNs).

            Prints the count of removed rows if any cleaning actions were performed.
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        consistent_rows = len(self.df)
        if initial_rows > consistent_rows:
            print(f"Cleaned {initial_rows - consistent_rows} rows (duplicates/NaN).")

    def visualize_distribution(self, target: str) -> None:
        """
            Plots and saves a histogram with a Kernel Density Estimate (KDE) for the target.

            Args:
                target (str): The name of the column to visualize.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[target], kde=True, color='blue')
        plt.title(f'Distribution of {target}')
        plt.xlabel('Count')
        plt.ylabel('Frequency')
        plt.savefig(DataframeUtils.DISTRIBUTION_PLOT_PATH)
        plt.show()

    def parse_date(self, date_column: str) -> None:
        """
            Extracts temporal features (Month, DayOfWeek) from a date column.

            The original date column is dropped after feature extraction.

            Args:
                date_column (str): The name of the column containing date strings.
        """
        self.df[date_column] = pd.to_datetime(self.df[date_column], format='%d/%m/%Y')
        self.df['Month'] = self.df[date_column].dt.month
        self.df['DayOfWeek'] = self.df[date_column].dt.dayofweek
        self.df = self.df.drop(columns=[date_column])

    def drop_unnecessary_data(self, column: str, standing_data_value: str) -> None:
        """
            Filters the dataset based on a specific column value and drops that column.

            Useful for removing rows where a business condition is not met
            (e.g., non-functioning days).

            Args:
                column (str): Column name to filter by.
                standing_data_value (str): The value to keep in the dataset.
        """
        self.df = self.df[self.df[column] == standing_data_value].copy()
        self.df = self.df.drop(columns=[column])

    def cyclical_encoding(self, column: str) -> None:
        """
            Applies sine and cosine transformations to periodic features.

            Ensures the model understands that values like 'Hour 23' and 'Hour 0' are close.

            Args:
                column (str): The name of the periodic column (e.g., 'Hour', 'Month').
        """
        if column in self.df.columns:
            max_val = self.df[column].max() + 1
            self.df[f'{column}_sin'] = (
                np.sin(2 * np.pi * (self.df[column]) / max_val))
            self.df[f'{column}_cos'] = (
                np.cos(2 * np.pi * (self.df[column]) / max_val))
            self.df = self.df.drop(column, axis=1)

    def one_hot_encode(self, columns: list) -> None:
        """
            Converts categorical variables into dummy/indicator variables.

            Uses drop_first=True to avoid the dummy variable trap.

            Args:
                columns (list): List of categorical columns to encode.
        """
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True, dtype=int)

    def build_heatmap(self) -> None:
        """
            Generates and saves a correlation heatmap for all numeric features.
        """
        plt.figure(figsize=(16, 12))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(DataframeUtils.HEATMAP_PATH)
        plt.show()

    def drop_highly_correlated_features(self, threshold: float = 0.9) -> None:
        """
            Identifies and removes features with high multicollinearity.

            Keeps only one feature from pairs that have a correlation coefficient above the threshold.

            Args:
                threshold (float): Correlation threshold (0 to 1). Defaults to 0.9.
        """
        corr_matrix = self.df.corr().abs()

        upper = corr_matrix.where(
                                np.triu(
                                    np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Deleted columns (threshold > {threshold}): {to_drop}")
        self.df = self.df.drop(columns=to_drop)

    def get_feature_names(self, target: str) -> list:
        """
            Retrieves the list of feature column names excluding the target.

            Args:
                target (str): The name of the target variable.

            Returns:
                list: Column names representing features.
        """
        df = self.df.copy()
        return df.drop(columns=[target]).columns.tolist()
