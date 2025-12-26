import os
import sys

from dataclasses import dataclass

import pandas as pd

from src.exception_handeling import CustomException
from src.Logger import logging
from sklearn.model_selection import train_test_split
from src.camponents.data_transformation import Datatransformation
from src.camponents.data_transformation import DataTransformationconfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_confic=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion methord or component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read The dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_confic.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_confic.raw_data_path,index=False,header=True)
            logging.info('Train Test Split')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
           
            train_set.to_csv(self.ingestion_confic.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_confic.test_data_path,index=False,header=True)

            logging.info('Inmgestion of data is completed ')
            return(
                self.ingestion_confic.train_data_path,
                self.ingestion_confic.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=='__main__':
    obj =DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = Datatransformation()
    data_transformation.initiate_data_tranformation(train_data,test_data)
       
 