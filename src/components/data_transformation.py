import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            onehot_columns = ['location']
            standard_columns=['total_sqft','bhk','bath']
            cat_pipeline=Pipeline(
                steps=[
                ("one_hot_encoder",OneHotEncoder(drop='first', sparse_output=False))
                ]
            )
            stan_pipeline=Pipeline(
                steps=[
                    ("Standard",StandardScaler())
                ]
            )

            logging.info(f"Columns: {onehot_columns}")

            preprocessor=ColumnTransformer(
                [             
                ("cat_pipelines",cat_pipeline,onehot_columns),
                ("stan_pipeline",stan_pipeline,standard_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df=train_df.drop(columns=["price"],axis=1)
            target_feature_train_df=train_df["price"]

            input_feature_test_df=test_df.drop(columns=["price"],axis=1)
            target_feature_test_df=test_df["price"]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            target_feature_train_df=np.array(target_feature_train_df)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

