import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Initialize paths to model and preprocessor
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = load_object(file_path=self.model_path)
        self.preprocessor = load_object(file_path=self.preprocessor_path)

    def predict(self, features):
        try:
            # Apply preprocessor to the input features
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, total_sqft: float, bath: float, bhk: int, location: str):
        self.total_sqft = total_sqft
        self.bath = bath
        self.bhk = bhk
        self.location = location

    def get_data_as_data_frame(self):
        try:
            # Construct DataFrame with expected column names
            custom_data_input_dict = {
                "total_sqft": [self.total_sqft],  # Match column names expected by preprocessor
                "bath": [self.bath],
                "bhk": [self.bhk],
                "location": [self.location]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
