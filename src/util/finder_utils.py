from dataclasses import dataclass


@dataclass
class FinderUtils:
    """
        Configuration utility for data source identification and ingestion.

        This class centralizes the parameters required to locate the raw dataset
        and defines the specific encoding strategy needed to parse the file correctly.
    """

    # The relative directory path where the raw CSV files are stored
    DATASET_FOLDER = 'source_data'

    # The specific filename of the Seoul Bike Sharing Demand dataset
    DATASET_NAME = 'SeoulBikeData.csv'

    # The encoding format required to read the dataset (unicode_escape is used
    # to handle special characters or non-standard formatting in the raw file)
    ENCODING = 'unicode_escape'
