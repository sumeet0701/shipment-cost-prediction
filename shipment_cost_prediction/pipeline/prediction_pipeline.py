from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.entity.artifact_entity import *
from shipment_cost_prediction.entity.config_entity import *
from shipment_cost_prediction.entity.raw_data_validation import IngestedDataValidation
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.config.configuration import Configuration
from shipment_cost_prediction.utils.utils import load_object
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.utils.utils import save_data
from tkinter import E

import numpy as np
import pandas as pd
import os
import sys
import shutil

class Prediction_Pipeline:
    def __init__(self,config:Configuration = Configuration()):
        """
        Prediction Class: It helps in predicting from saved model.
                          It has two modes Bulk prediction and single prediction
            created by :
                    Sumeet Maheshwari
        """

        logging.info(f"\n{'*'*20} Prediction Pipeline Initiated {'*'*20}\n")

        # getting data validation config info.
        self.data_validation_config = config.get_data_validation_config()

        # loading Feature Engineering, Preprocessing and Model pickle object for prediction
        self.fe_obj = load_object(file_path= os.path.join(ROOT_DIR,
                                                          PIKLE_FOLDER_NAME_KEY,
                                                          "feat_eng.pkl"))
        self.preprocessing_obj = load_object(file_path= os.path.join(ROOT_DIR,
                                                                     PIKLE_FOLDER_NAME_KEY,
                                                                     "preprocessed.pkl"))
        self.model_obj = load_object(file_path= os.path.join(ROOT_DIR,
                                                             PIKLE_FOLDER_NAME_KEY,
                                                             "model.pkl"))
        
        # Reading schema.yaml file to validate prediction data
        self.schema_file_path = self.data_validation_config.schema_file_path
        self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)

    
    def initiate_bulk_predictions(self):

        """
        Function to predict from saved trained model for entire dataset. It returns the original dataset \n
        with prediction column
        """
        try:
            logging.info(f"{'*'*20}Bulk Prediction Mode Selected {'*'*20}")
            # Getting location of uploaded dataset
            self.folder = PREDICTION_DATA_SAVING_FOLDER_KEY
            self.path = os.path.join(self.folder,os.listdir(self.folder)[0])

            # validating Uploaded dataset
            logging.info(f"Validatiog Passed Dataset : [{self.path}]")
            pred_val = IngestedDataValidation(self.path,self.data_validation_config)
            data_validation_status = pred_val.validate_dataset_schema()
            
            
            logging.info(f"Prediction for dataset: [{self.path}]")

            if data_validation_status:
                # reading uploaded csv file in pandas
                data_df = pd.read_csv(self.path)
                columns = ['ID', 'Project_Code', 'PQ', 'PO/SO', 'ASN/DN', 'Country', 'Managed_By',
                            'Fulfill_Via', 'Vendor_INCO_Term', 'Shipment_Mode',
                            'PQ_First_Sent_to_Client_Date', 'PO_Sent_to_Vendor_Date',
                            'Scheduled_Delivery_Date', 'Delivered_to_Client_Date',
                            'Delivery_Recorded_Date', 'Product_Group', 'Sub_Classification',
                            'Vendor', 'Item_Description', 'Molecule/Test_Type', 'Brand', 'Dosage',
                            'Dosage_Form', 'Unit_of_Measure_(Per_Pack)', 'Line_Item_Quantity',
                            'Line_Item_Value', 'Pack_Price', 'Unit_Price', 'Manufacturing_Site',
                            'First_Line_Designation', 'Weight_(Kilograms)', 'Freight_Cost_(USD)',
                            'Line_Item_Insurance_(USD)']
                
                logging.info("Feature Engineering applied !!!")
                featured_eng_data = pd.DataFrame(self.fe_obj.transform(data_df),columns=columns)
                featured_eng_data.drop(columns="Freight_Cost_(USD)", inplace=True)
                logging.info("Data Preprocessing Done!!!")
                # Applying preprocessing object on the data
                transformed_data = pd.DataFrame(np.c_[self.preprocessing_obj.transform(featured_eng_data)],columns=columns)
                # Convertng datatype of feature accordingly
                transformed_data=transformed_data.infer_objects()

                # Predicting from the saved model object
                prediction = self.model_obj.predict(transformed_data)
                data_df["predicted_demand"] = prediction
                logging.info("Prediction from model done")

                logging.info("Saving prediction file for sending it to the user")

                output_folder_file_path = os.path.join(ROOT_DIR,"Output Folder",CURRENT_TIME_STAMP,"Predicted.csv")
                if os.path.exists(os.path.join(ROOT_DIR,"Output Folder")):
                    shutil.rmtree(os.path.join(ROOT_DIR,"Output Folder"))

                save_data(file_path=output_folder_file_path,data = data_df)
                zipped_file = os.path.dirname(output_folder_file_path)
                shutil.make_archive(zipped_file,"zip",zipped_file)
                shutil.rmtree(zipped_file)
                shutil.rmtree(self.folder)
                
                logging.info(f"{'*'*20} Bulk Prediction Coomplete {'*'*20}")
                return zipped_file+".zip"
                   
        except Exception as e:
            raise CustomException(e,sys) from e
        
    
    def initiate_single_prediction(self,data:dict)-> int:
        """
        Function to predict from the saved train model. It predicts from single value of each feature.
        """
        try:
            logging.info(f"{'*'*20} Single Prediction Mode Selected {'*'*20}")
            logging.info(f"Passed Info: [{data}]")

            # Converting passed data into DataFrame
            df = pd.DataFrame([data])

            
            # Applying preprocessing Object On the data
            preprocessed_df = pd.DataFrame(np.c_[self.preprocessing_obj.transform(df)], columns= df.columns())
            # Changing datatype of features accordingly
            preprocessed_df = preprocessed_df.infer_objects()

            # Predicting from the saved model
            prediction = self.model_obj.predict(preprocessed_df)
            logging.info(f"{'*'*20} Single Prediction Complete {'*'*20}")
            return round(prediction[0])
        
        except Exception as e:
            raise CustomException(e,sys) from e