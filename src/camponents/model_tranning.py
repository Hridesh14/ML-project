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
            params = {
    'Decision_tree': {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_features': ['sqrt', 'log2']
    },
    'Randomforest': {
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth': [None, 5, 10, 15],
        'max_features': ['sqrt', 'log2', None]
    },
    'Grandient Boosting': {
        'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'criterion': ['squared_error', 'friedman_mse'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    'Linear Regression': {
        # Linear Regression has no major hyperparameters to tune, 
        # but you can toggle fit_intercept
        'fit_intercept': [True, False]
    },
    'K-Neighbors Classifier': {
        'n_neighbors': [5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    },
    'XGBclassifier': {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256],
        'max_depth': [3, 5, 6, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
    },
    'CatBoosting': {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    'AdaBoost': {
        'learning_rate': [0.1, 0.01, 0.5, 0.001],
        'loss': ['linear', 'square', 'exponential'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}

            

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models,params=params)
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
        

