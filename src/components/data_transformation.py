import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_transformer_object(self):
        
        # The function is responsible for data transformation.
        # It creates a preprocessor object that will be used to transform the data.
        
        try:
            num_col = ['reading_score', 'writing_score']
            cat_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns Standard Scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_col),
                 ("cat_pipeline", cat_pipeline, cat_col)]
            )
            return preprocessor
            
        except Exception as es:
            raise CustomException(es, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_transformer_object()

            target_column = "math_score"
            num_columns = ['reading_score', 'writing_score']

            input_feature_train_data = train_data.drop(columns = [target_column], axis = 1)
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop(columns = [target_column], axis = 1)
            target_feature_test_data = test_data[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_data) ]
            test_arr = np.c_[ input_feature_test_arr, np.array(target_feature_test_data) ]

            logging.info('Saved preprocessing object.')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,   test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)