import os  
import sys 
import json
import pandas as pd
import shutil

from shipment_cost_prediction.config import *
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.entity.config_entity import *
from shipment_cost_prediction.entity.artifact_entity import *
from shipment_cost_prediction.config import configuration
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.entity.raw_data_validation import IngestedDataValidation

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n")           
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_path = self.data_validation_config.schema_file_path
            self.train_data = IngestedDataValidation(
                validate_path=self.data_ingestion_artifact.train_file_path, schema_path=self.schema_path)
            self.test_data = IngestedDataValidation(
                validate_path=self.data_ingestion_artifact.test_file_path, schema_path=self.schema_path)
            
            self.train_path = self.data_ingestion_artifact.train_file_path
            self.test_path = self.data_ingestion_artifact.test_file_path
            
            self.validated_train_path = self.data_validation_config.validated_train_path
            self.validated_test_path =self.data_validation_config.validated_test_path
        
        except Exception as e:
            raise CustomException(e,sys) from e


    def isFolderPathAvailable(self) -> bool:
        try:

             # True means avaliable false means not avaliable
             
            isfolder_available = False
            train_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path
            if os.path.exists(train_path):
                if os.path.exists(test_path):
                    isfolder_available = True
            return isfolder_available
        except Exception as e:
            raise CustomException(e, sys) from e     
      


        
    def is_Validation_successfull(self):
        try:
            validation_status = True
            logging.info("Validation Process Started")
            if self.isFolderPathAvailable() == True:
                train_filename = os.path.basename(
                    self.data_ingestion_artifact.train_file_path)

                is_train_filename_validated = self.train_data.validate_filename(
                    file_name=train_filename)

                is_train_column_name_same = self.train_data.check_column_names()

                is_train_missing_values_whole_column = self.train_data.missing_values_whole_column()

                self.train_data.replace_null_values_with_null()

                test_filename = os.path.basename(
                    self.data_ingestion_artifact.test_file_path)

                is_test_filename_validated = self.test_data.validate_filename(
                    file_name=test_filename)

                is_test_column_name_same = self.test_data.check_column_names()

                is_test_missing_values_whole_column = self.test_data.missing_values_whole_column()

                self.test_data.replace_null_values_with_null()

                logging.info(
                    f"Train_set status|is Train filename validated?: {is_train_filename_validated}|is train column name validated?: {is_train_column_name_same}|whole missing columns?{is_train_missing_values_whole_column}")
                logging.info(
                    f"Test_set status|is Test filename validated?: {is_test_filename_validated}|is test column names validated? {is_test_column_name_same}| whole missing columns? {is_test_missing_values_whole_column}")

                if is_train_filename_validated  & is_train_column_name_same & is_train_missing_values_whole_column:
                    ## Exporting Train.csv file 
                    # Create the directory if it doesn't exist
                    os.makedirs(self.validated_train_path, exist_ok=True)

                    # Copy the CSV file to the validated train path
                    shutil.copy(self.train_path, self.validated_train_path)
                    self.validated_train_path=os.path.join(self.validated_train_path,FILE_NAME)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated train dataset to file: [{self.validated_train_path}]")
                                     
                                     
                                        
                    ## Exporting test.csv file
                    os.makedirs(self.validated_test_path, exist_ok=True)
                    logging.info(f"Exporting validated train dataset to file: [{self.validated_train_path}]")
                    os.makedirs(self.validated_test_path, exist_ok=True)
                    # Copy the CSV file to the validated train path
                    shutil.copy(self.test_path, self.validated_test_path)
                    self.validated_test_path=os.path.join(self.validated_test_path,FILE_NAME)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated test dataset to file: [{self.validated_test_path}]")
                                        
                    
                    return validation_status,self.validated_train_path,self.validated_test_path
                else:
                    validation_status = False
                    logging.info("Check yout Training Data! Validation Failed")
                    raise ValueError(
                        "Check your Training data! Validation failed")
                

            return validation_status,"NONE","NONE"
        except Exception as e:
                raise CustomException(e, sys) from e      
        """
            else:
                    validation_status = False
                    logging.info("Check yout Training Data! Validation Failed")
                    raise ValueError(
                        "Check your Training data! Validation failed")

                if is_test_filename_validated  & is_test_column_name_same & is_test_missing_values_whole_column:
                    pass
                else:
                    validation_status = False
                    logging.info("Check your Test data! Validation failed")
                    raise ValueError(
                        "Check your Testing data! Validation failed")

                logging.info("Validation Process Completed")

                return validation_status

        except Exception as e:
            raise CustomException(e, sys) from e      
    """
    def get_train_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise CustomException(e,sys) from e
    def get_and_save_data_drift_report(self):
        try:
            logging.info("Generating data drift report.json file")
            profile = Profile(sections = [DataDriftProfileSection()])
            train_df, test_df = self.get_train_test_df()
            profile.calculate(train_df, test_df)
            
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent = 6)
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise CustomException(e,sys) from e   

    def save_data_drift_report_page(self):
        try:
            logging.info("Generating data drift report.html page")
            dashboard = Dashboard(tabs = [DataDriftTab()])
            train_df, test_df = self.get_train_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
            logging.info("Report.html page generation successful!!")
        except Exception as e:
            raise CustomException(e,sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            logging.info("Checking for Data Drift")
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise CustomException(e,sys) from e
    
    

    def initiate_data_validation(self):
        try:
            #self.validate_dataset_schema()
            #self.is_train_test_file_exists()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.schema_path, 
                is_validated=self.is_Validation_successfull(),
                validated_train_path = self.validated_train_path,
                validated_test_path= self.validated_test_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                message="Data validation performed"
            )
            logging.info(
                f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30}")