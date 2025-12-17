from sklearn.preprocessing import StandardScaler

from finder.data_finder import DataFinder
from model.data_normalizer import DataNormalizer
from model.data_producer import DataProducer
from model.predictor import Predictor
from util.dataframe_utils import DataframeUtils
from util.finder_utils import FinderUtils
from util.predictor_utils import PredictorUtils


def main():
    finder = DataFinder(FinderUtils.DATASET_FOLDER,
                        FinderUtils.DATASET_NAME,
                        FinderUtils.ENCODING)
    df = finder.get_data()
    df.columns = DataframeUtils.BIKE_DATA_COLUMNS
    producer = DataProducer(df)
    producer.parse_date('Date')
    producer.drop_unnecessary_data('Functioning_Day', 'Yes')
    producer.one_hot_encode(DataframeUtils.NON_NUMERIC_COLUMNS)
    producer.build_heatmap()
    producer.drop_highly_correlated_features(DataframeUtils.FEATURES_CORRELATION_VALUE)
    producer.cyclical_encoding('Month')
    producer.cyclical_encoding('Hour')

    normalizer = DataNormalizer(StandardScaler())
    predictor = Predictor(producer.df, normalizer, PredictorUtils.MODELS)
    df = predictor.predict()
    print(df)

if __name__ == '__main__':
    main()
