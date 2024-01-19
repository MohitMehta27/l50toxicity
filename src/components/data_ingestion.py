import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def handle_outliers_iqr(self, column):
        Q1 = np.percentile(column, 25)
        Q3 = np.percentile(column, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(column, lower_bound, upper_bound)

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.read_csv(os.path.join(r'C:/LC50_toxicity_prediction/qsar.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=30)

            columns_to_handle_outliers = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']

            for column in columns_to_handle_outliers:
                train_set[column] = self.handle_outliers_iqr(train_set[column])
                test_set[column] = self.handle_outliers_iqr(test_set[column])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)
        


'''
Wrote this to check code is running successfully

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print("Train Data Path:", train_data_path)
    print("Test Data Path:", test_data_path)
'''