import ast
import logging
import sys
import time
import os
import pandas as pd

from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.utils.utils import save_object
from shipment_cost_prediction.entity.config_entity import ModelTrainerConfig
from shipment_cost_prediction.entity.artifact_entity import DataTransformationArtifact
from shipment_cost_prediction.entity.artifact_entity import ModelTrainerArtifact
from shipment_cost_prediction.constant import *
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from skopt import BayesSearchCV
from matplotlib import pyplot as plt

import sys
import os
import pandas as pd
import time
import joblib


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'*'*20} Model Training started {'*'*20}\n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.results = []
            # Define the models and their hyperparameters
            self.models = {
                "Linear Regression": [LinearRegression(), {'fit_intercept ': [True, False],
                                                            'normalize ': [True, False],
                                                            "copy_X": [True, False]}],
                "Random Forest": [RandomForestRegressor(), {'max_depth': [15, 20, 25, 30, 40],
                                                             'n_estimators': [100,150,200]}],
                "XG_Boost": [XGBRegressor(eval_metric='rmsle'),
                             {"max_depth": [12,15,20,25],
                              "n_estimators": [50, 150,200], "learning_rate": [0.05,0.1,0.150,0.2,0.25]}]
                              }
        except Exception as e:
            raise CustomException(e, sys)

    def run_grid_search(self, X_train, y_train, X_test, y_test):
        # Define a custom scorer that scales R-squared scores between 0-100
        r2_scorer = make_scorer(lambda y_true, y_pred: r2_score(y_true, y_pred) * 100)

        # Loop through the models dictionary and fit each model with GridSearchCV
        for model_name, (model, param_grid) in self.models.items():
            logging.info(f"Fitting {model_name}...")
            print(f"Model selected: {model_name}")

            # Fit the model with GridSearchCV using the custom scorer
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring=r2_scorer, cv=5)
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            elapsed_time_secs = time.time() - start_time
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_secs))

            # Record the results of GridSearchCV
            best_params = grid_search.best_params_
            r2 = grid_search.best_score_  # already scaled between 0-100 by the custom scorer
            mse = metrics.mean_squared_error(y_test, grid_search.predict(X_test))
            n = X_test.shape[0]
            k = len(best_params)

            # Get feature importances if applicable
            try:
                importances = model.feature_importances_
            except AttributeError:
                importances = None

            # Add the results to the list as a dictionary
            result_dict = {'Model': model_name, 'Best Parameters': str(best_params),
                        'Mean Squared Error': mse, 'R2 Score': r2, 'Time Taken': elapsed_time_str,
                        'Feature Importances': importances}
            self.results.append(result_dict)

            # Plot and save the feature importance if applicable
            if importances is not None:
                plt.bar(range(len(importances)), importances)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title(f'{model_name} Feature Importance')
                # save the plot in the specified directory
                # save the plot in the specified directory
                file_name = f"{model_name}_feature_importance.png"
                file_path = os.path.join("C:", os.sep, "Users", "Admin", "Documents", "Shipment_pricing_Project", "prediction_files", file_name)

                plt.savefig(file_path)
            # Print the results for this model
            logging.info(f"******Results for********* {model_name}:")
            logging.info(f"Best hyperparameters: {best_params}")
            logging.info(f"Mean squared error: {mse:.4f}")
            logging.info(f"R2 score: {r2:.4f}")
            if importances is not None:
                logging.info(f"Feature importances: {importances}")
            logging.info(f"Time taken: {elapsed_time_str}\n")

        # Create a DataFrame from the list of dicts
        self.results_df = pd.DataFrame(self.results, columns=result_dict.keys())
        
        
        
    def get_best_model(self, X_train, y_train, X_test, y_test):
        
        self.run_grid_search(X_train, y_train, X_test, y_test)
        # Sort the results by the R-squared score in descending order
        sorted_results = self.results_df.sort_values('R2 Score', ascending=False)

        # Choose the model with the highest R-squared score
        best_model_name = sorted_results.iloc[0]['Model']
        params_dict = ast.literal_eval(sorted_results.iloc[0]['Best Parameters'])
        best_model = self.models[best_model_name][0].set_params(**params_dict)

        logging.info(f"Fitting {best_model_name}...")
        best_model.fit(X_train, y_train)

        logging.info(f"The best model based on R-squared score is {best_model_name}.")
        return best_model

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training and Test")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Transformed Data found!!! Now, converting it into dataframe")
            train_df = pd.read_csv(transformed_train_file_path)
            test_df = pd.read_csv(transformed_test_file_path)

            target_column_name = 'Freight_Cost_USD_Clean'

            logging.info("Splitting Input features and Target Feature for train and test data")
            train_target_feature = train_df[target_column_name]
            train_input_feature = train_df.drop(columns=[target_column_name], axis=1)

            test_target_feature = test_df[target_column_name]
            test_input_feature = test_df.drop(columns=[target_column_name], axis=1)

            logging.info("Best Model Finder function called")
            model_obj = self.get_best_model(train_input_feature, train_target_feature, test_input_feature,
                                            test_target_feature)
            
            logging.info("Saving best model object file")
            trained_model_object_file_path = self.model_trainer_config.trained_model_file_path
            save_object(file_path=trained_model_object_file_path, obj=model_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(trained_model_object_file_path)),obj=model_obj)


            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, 
                                                          message="Model Training Done!!",
                                                          trained_model_object_file_path=trained_model_object_file_path)
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Model Training log completed {'*'*20}\n\n")