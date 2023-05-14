from shipment_cost_prediction.components import data_validation
from shipment_cost_prediction.config.configuration import Configuration
from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.entity.raw_data_validation import IngestedDataValidation
from shipment_cost_prediction.utils.utils import load_object
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.utils.utils import save_data
import os, sys
import shutil
import pandas as pd
import numpy as np

class Prediction:
    def __init__(self,
                 config:Configuration = Configuration()):
        """Prediction Class : It helps in predicting from saved trained model.
                           It has two modes Bulk Prediction and Single Prediction
        """
        logging.info(f"\n{'*'*20} Prediction Pipeline Initiated {'*'*20}\n")

         # Getting data validation config info
        self.data_validation_config = config.get_data_validation_config()
        
        # Loading Feature Engineering, Preprocessing and Model pickle objects for prediction
        self.fe_obj = load_object(file_path=os.path.join(
            ROOT_DIR,
            PIKLE_FOLDER_NAME_KEY,"feat_eng.pkl"))
        

        self.preprocessing_obj = load_object(file_path=os.path.join(
            ROOT_DIR,
            PIKLE_FOLDER_NAME_KEY,"preprocessed.pkl"))
        

        self.model_obj = load_object(file_path=os.path.join(
            ROOT_DIR,
            PIKLE_FOLDER_NAME_KEY,"model.pkl"))
        

        # Reading schema.yaml file to validate prediction data
        self.schema_file_path = self.data_validation_config.schema_file_path
        self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)

    
    def initiate_bulk_prediction(self):
        """
        Function to predict from saved trained model for entire dataset. It returns the original dataset \n
        with prediction column
        """
         
        try:
            logging.info(f"{'*'*20}Bulk Prediction Mode Selected {'*'*20}")
            # Getting location of uploaded dataset
            self.folder = PREDICTION_DATA_SAVING_FOLDER_KEY
            self.path = os.path.join(self.folder,os.listdir(self.folder)[0])

            

        except Exception as e:
            raise CustomException(e,sys) from e
