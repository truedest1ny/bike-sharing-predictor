import pandas as pd
from pathlib import Path

class DataFinder:
    def __init__(self, folder:str, filename:str, encoding_strategy:str):
        self.folder = folder
        self.filename = filename
        self.encoding_strategy = encoding_strategy


    def get_data(self) -> pd.DataFrame:
        base_path = Path('.').resolve()
        csv_path = base_path / self.folder / self.filename

        if not csv_path.exists():
            csv_path = Path(__file__).resolve().parents[2] / self.folder / self.filename

        if not csv_path.exists():
            raise FileNotFoundError(f"File was not found on path: {csv_path.absolute()}")

        return pd.read_csv(csv_path, encoding=self.encoding_strategy)
