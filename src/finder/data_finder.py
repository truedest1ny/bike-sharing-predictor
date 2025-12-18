import pandas as pd
from pathlib import Path


class DataFinder:
    """
        Handles data ingestion by locating and loading CSV files.

        This class implements a robust file-searching strategy, checking multiple
        directory levels to ensure the dataset is found regardless of the
        execution context (e.g., running from terminal vs. IDE).
    """
    def __init__(self, folder:str, filename:str, encoding_strategy:str):
        """
            Initializes the DataFinder with path and encoding details.

            Args:
                folder (str): The name of the directory where the dataset is stored.
                filename (str): The name of the CSV file (including extension).
                encoding_strategy (str): The character encoding used to read the file (e.g., 'utf-8', 'latin1').
        """
        self.folder = folder
        self.filename = filename
        self.encoding_strategy = encoding_strategy

    def get_data(self) -> pd.DataFrame:
        """
            Locates the file and loads it into a Pandas DataFrame.

            The method attempts to resolve the path relative to the current working
            directory first, then falls back to a path relative to the script location.

            Returns:
                pd.DataFrame: The loaded dataset.

            Raises:
                FileNotFoundError: If the file cannot be located in the expected directories.
         """
        base_path = Path('.').resolve()
        csv_path = base_path / self.folder / self.filename

        if not csv_path.exists():
            csv_path = Path(__file__).resolve().parents[2] / self.folder / self.filename

        if not csv_path.exists():
            raise FileNotFoundError(f"File was not found on path: {csv_path.absolute()}")

        return pd.read_csv(csv_path, encoding=self.encoding_strategy)
