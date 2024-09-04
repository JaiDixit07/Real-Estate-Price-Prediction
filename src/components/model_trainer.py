import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str= os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        logging.info("Entered the Model Training method.")
        try:    
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": xgb.XGBRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info("Ingestion of the data is completed")

            best_model=models[best_model_name]

            

            # Define the hyperparameter grid for Random Forest

            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            logging.info("Performing Randomized Search with Cross-Validation for Random Forest")
            rf = RandomForestRegressor()
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
            rf_random.fit(X_train, y_train)

            best_params = rf_random.best_params_
            logging.info(f"Best Parameters found: {best_params}")

            # Train the final model with the best parameters
            best_rf = rf_random.best_estimator_
            best_rf.fit(X_train, y_train)

            # Evaluate the model
            y_pred = best_rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Test RMSE: {rmse:.4f}")
            logging.info(f"Test R²: {r2:.4f}")

            # Ensemble Model
            logging.info("Creating and evaluating ensemble model")
            model1 = RandomForestRegressor(**best_params, random_state=42)
            model2 = GradientBoostingRegressor(random_state=42)
            model3 = xgb.XGBRegressor(random_state=42)

            ensemble_model = VotingRegressor(estimators=[('rf', model1), ('gb', model2), ('xgb', model3)])
            ensemble_model.fit(X_train, y_train)

            # Evaluate the ensemble model
            y_pred_ensemble = ensemble_model.predict(X_test)
            test_rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            test_r2_ensemble = r2_score(y_test, y_pred_ensemble)
            logging.info(f"Ensemble Test RMSE: {test_rmse_ensemble:.4f}")
            logging.info(f"Ensemble Test R²: {test_r2_ensemble:.4f}")

            # Save the best model (could be the ensemble or best_rf based on results)
            if test_r2_ensemble > r2:
                logging.info("Ensemble model outperformed individual best model")
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=ensemble_model)
                return test_r2_ensemble
            else:
                logging.info("Individual best model outperformed ensemble model")
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_rf)
                return r2


        except Exception as e:
            raise CustomException(e, sys)
