import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, MinMaxScaler

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

    def get_data_transformer_object(self,df):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # Mean prices by location
            mean_prices = df.groupby('location')['price'].mean()

            # Map the mean prices to the original DataFrame
            df['location_encoded'] = df['location'].map(mean_prices)

            # Calculate the global mean of the target variable
            global_mean = df['price'].mean()

            # Define the smoothing parameter
            m = 3

            # Apply smoothing to the target encoding
            def smooth_mean_encoding(col, target, m, global_mean):
                agg = df.groupby(col)[target].agg(['mean', 'count'])
                mean_encoded = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
                return mean_encoded

            # Apply the smooth mean encoding to the 'standardized_location' column
            mean_encoded = smooth_mean_encoding('location', 'price', m, global_mean)
            df['location_encoded'] = df['location'].map(mean_encoded)

            # Additional preprocessing steps can be added here
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['location_encoded']),
                    # Add other transformers for different columns here
                ]
            ) 

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(train_df)

            # Mean encoding for the training data
            mean_prices = train_df.groupby('location')['price'].mean()
            train_df['location_encoded'] = train_df['location'].map(mean_prices)

            # Mean encoding for the test data using the training data's mean
            test_df['location_encoded'] = test_df['location'].map(mean_prices)

            input_feature_train_df = train_df.drop(columns=["price"], axis=1)
            target_feature_train_df = train_df["price"]

            input_feature_test_df = test_df.drop(columns=["price"], axis=1)
            target_feature_test_df = test_df["price"]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
