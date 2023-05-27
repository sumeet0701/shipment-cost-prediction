from shipment_cost_prediction.config.configuration import Configuration
from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.entity.artifact_entity import DataIngestionArtifact
from shipment_cost_prediction.entity.artifact_entity import DataValidationArtifact
from shipment_cost_prediction.entity.artifact_entity import DataTransformationArtifact
from shipment_cost_prediction.entity.artifact_entity import ModelTrainerArtifact
from shipment_cost_prediction.components.data_ingestion import DataIngestion
from shipment_cost_prediction.components.data_validation import DataValidation
from shipment_cost_prediction.components.data_transformation import DataTransformation
from shipment_cost_prediction.components.model_trainer import ModelTrainer

from multiprocessing import Process
from threading import Thread
from typing import List
from collections import namedtuple
from datetime import datetime

import uuid
import os, sys
import pandas as pd



class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) from e


    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                       data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.config.get_data_transformation_config(),
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def start_model_training(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                        data_transformation_artifact=data_transformation_artifact)   

            return model_trainer.initiate_model_training()
        except Exception as e:
            raise CustomException(e,sys) from e  



    
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                             data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)  

         
        except Exception as e:
            raise CustomException(e, sys) from e