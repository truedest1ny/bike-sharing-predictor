from finder.data_finder import DataFinder
from producer.data_producer import DataProducer
from util.dataframe_utils import DataframeUtils
from util.finder_utils import FinderUtils


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

if __name__ == '__main__':
    main()
