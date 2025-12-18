from sklearn.preprocessing import StandardScaler

from finder.data_finder import DataFinder
from model.data_normalizer import DataNormalizer
from model.data_producer import DataProducer
from model.predictor import Predictor
from model.visualizer import Visualizer
from util.dataframe_utils import DataframeUtils
from util.finder_utils import FinderUtils
from util.predictor_utils import PredictorUtils


class Starter:
    """
        Service class to orchestrate the entire Machine Learning pipeline.

        This class handles the flow from raw data ingestion and preprocessing
        to model training, evaluation, and result visualization.
    """
    @staticmethod
    def start() -> None:
        """
            Entry point to execute the full pipeline.

            Orchestrates the preprocessing of data followed by the machine learning
            process, including model comparison and interpretation.
        """
        # Step 1: Ingest and prepare data for modeling
        producer = Starter._data_preprocessing()

        # Step 2: Run ML experiments and visualize outcomes
        Starter._machine_learning(producer)

    @staticmethod
    def _data_preprocessing() -> DataProducer:
        """
            Handles the Exploratory Data Analysis (EDA) and data cleaning phase.

            Returns:
                DataProducer: An object containing the processed and encoded DataFrame.
        """
        # Initialize data finder to locate and load the dataset
        finder = DataFinder(FinderUtils.DATASET_FOLDER,
                            FinderUtils.DATASET_NAME,
                            FinderUtils.ENCODING)
        df = finder.get_data()

        # Map raw columns to standardized names defined in utils
        df.columns = DataframeUtils.BIKE_DATA_COLUMNS

        producer = DataProducer(df)

        # Data Cleaning: Handle missing values and duplicates
        producer.clean_data()

        # Visual EDA: Understand target distribution
        producer.visualize_distribution(DataframeUtils.TARGET_COLUMN)

        # Convert dates and filter operational days
        producer.parse_date('Date')
        producer.drop_unnecessary_data('Functioning_Day', 'Yes')

        # Encoding: Convert categorical variables to numeric format
        producer.one_hot_encode(DataframeUtils.NON_NUMERIC_COLUMNS)

        # Statistical Analysis: Heatmap for multicollinearity detection
        producer.build_heatmap()
        producer.drop_highly_correlated_features(DataframeUtils.FEATURES_CORRELATION_VALUE)

        # Cyclical Encoding: Handle periodic features (Month/Hour) using Sin/Cos transforms
        producer.cyclical_encoding('Month')
        producer.cyclical_encoding('Hour')

        return producer

    @staticmethod
    def _machine_learning(producer: DataProducer) -> None:
        """
            Executes model training, evaluation, and visualization of results.

            Args:
                producer (DataProducer): The object containing preprocessed training data.
        """
        # Feature Scaling: Apply standardization to normalize feature ranges
        normalizer = DataNormalizer(StandardScaler())

        # Predictor: Handles Train/Test split and trains multiple regression models
        predictor = Predictor(producer.df, normalizer, PredictorUtils.MODELS)
        metrics_df = predictor.predict()

        # Compare model performance metrics (MAE, RMSE, R2)
        Visualizer.visualize_comparing(metrics_df)

        # Interpretation: Select the best performing model based on R-squared
        best_model_name = Predictor.define_best_model(metrics_df, 'R2')
        best_model = PredictorUtils.MODELS[best_model_name]

        # Get feature names for importance analysis (ensuring correct order)
        feature_names = producer.get_feature_names(PredictorUtils.TARGET_COLUMN)

        # Diagnostics: Visualize what the model learned and where it fails
        Visualizer.plot_feature_importance(best_model, feature_names)
        Visualizer.plot_regression_results(
            predictor.y_test, predictor.predictions[best_model_name], best_model_name)
