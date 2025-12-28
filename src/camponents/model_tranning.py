import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception_handeling import CustomException
from src.Logger import logging

from src.utils import save_object


@dataclass
class Modeltranningconfig:
    trained_model_file_pathst= os.path.join('artifacts','modelTranner.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltranningconfig=Modeltranningconfig()
    

    def initate_model_trainner(self,train_array,test_array):
        try:
            logging.info('Spliting training and test input data')
            x_train,y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models ={
                'Randomforest':RandomForestRegressor(),
                'Decision_tree':DecisionTreeRegressor(),
                'Grandient Boosting':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'K-Neighbors Classifier':KNeighborsRegressor(),
                'XGBclassifier':XGBRegressor(),
                'CatBoosting': CatBoostRegressor(verbose=False),
                'AdaBoost':AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('no best model found')
            logging.info(f'Best found model on both tranning and testing dataset')

            save_object(
                file_path=self.modeltranningconfig.trained_model_file_pathst,
                obj=best_model
            )
            predicted =best_model.predict(x_test)
            r2_qure = r2_score(y_test,predicted)
            return r2_qure


        except Exception as e:
            raise CustomException(e,sys)
        

