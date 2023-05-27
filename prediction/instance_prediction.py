from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.entity.artifact_entity import ModelTrainerArtifact
from shipment_cost_prediction.entity.artifact_entity import DataTransformationArtifact

from sklearn.pipeline import Pipeline

import sys 
import pandas as pd
import pickle
import os
import logging

BATCH_PREDICTION = "batch_prediction"
INSTANCE_PREDICTION="Instance_prediction"
input_file_path="SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path ="prediction_files/feat_eng.pkl"
transformer_file_path ="prediction_files/preprocessed.pkl"
modmodel_file_pathel ="prediction_files/model.pkl"






import pandas as pd
import joblib

# Load the preprocessor and machine learning model
preprocessor = joblib.load('prediction_files/preprocessed.pkl')
model = joblib.load('prediction_files/model.pkl')

# Define mappings
COUNTRY_MAP = {'Zambia': 0, 'Ethiopia': 1, 'Nigeria': 2, 'Tanzania': 3, "Côte d'Ivoire": 4, 'Mozambique': 5, 'Others': 6, 'Zimbabwe': 7, 'South Africa': 8, 'Rwanda': 9, 'Haiti': 10, 'Vietnam': 11, 'Uganda': 12}
FULFILL_VIA_MAP = {'From RDC': 0, 'Direct Drop': 1}
SHIPMENT_MODE_MAP = {'Truck': 0, 'Air': 1, 'Air Charter': 2, 'Ocean': 3}
SUB_CLASSIFICATION_MAP = {'Adult': 0, 'Pediatric': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'Malaria': 4, 'ACT': 5}
DOSAGE_FORM= {'Tablet': 0, 'Test kit': 1, 'Oral': 2, 'Capsule': 3}
BRAND_MAP = {'Generic': 0, 'Others': 1, 'Determine': 2, 'Uni-Gold': 3}
#FIRST_LINE_DESIGNATION_MAP = {'Yes': 0, 'No': 1}


class instance_prediction_class:
    def __init__(self,weight_kg, line_item_quantity,line_item_value,fulfill_via, shipment_mode, country, brand, sub_classification,dosage_form) -> None:
        self.weight_kg = weight_kg
        self.line_item_quantity = line_item_quantity
        self.line_item_value=line_item_value
        self.fulfill_via = fulfill_via
        self.shipment_mode = shipment_mode
        self.country = country
        self.brand = brand
        self.sub_classification = sub_classification
        self.dosage_form=dosage_form
        
    
    def preprocess_input(self,weight_kg, line_item_quantity,line_item_value,fulfill_via, shipment_mode, country, brand, sub_classification,dosage_form):
        # Convert categorical variables to numerical format
        fulfill_via = FULFILL_VIA_MAP[fulfill_via]
        shipment_mode = SHIPMENT_MODE_MAP[shipment_mode]
        country = COUNTRY_MAP[country]
        brand = BRAND_MAP[brand]
        sub_classification = SUB_CLASSIFICATION_MAP[sub_classification]
        dosage_form = DOSAGE_FORM[dosage_form]

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'Weight_Kilograms_Clean': [weight_kg],
            'Line_Item_Quantity': [line_item_quantity],
            'Line_Item_Value':[line_item_value],
            'Fulfill_Via': [fulfill_via],
            'Shipment_Mode': [shipment_mode],
            'Country': [country],
            'Brand': [brand],
            'Sub_Classification': [sub_classification],
            'Dosage_Form': [dosage_form]
        })

        # Preprocess the user input using the preprocessor
        preprocessed_input = preprocessor.transform(user_input)

        # Return the preprocessed input as a numpy array
        return preprocessed_input

    def predict_price(self,preprocessed_input):
        # Make a prediction using the pre-trained model
        predicted_price = model.predict(preprocessed_input)
        

        # Return the predicted shipment price
        return predicted_price[0]

    def predict_price_from_input(self):
        


        # Preprocess the input using the preprocessor
        preprocessed_input = self.preprocess_input(self.weight_kg, self.line_item_quantity,self.line_item_value,self.fulfill_via, self.shipment_mode, 
                                                   self.country, self.brand, self.sub_classification, self.dosage_form)

# Make a prediction using the pre-trained model
        predicted_price = self.predict_price(preprocessed_input)

        # Round off the predicted shipment price to two decimal places
        rounded_price = round(predicted_price, 2)
        predicted_price=rounded_price

        # Print the rounded predicted shipment price
        print("The predicted shipment price is: $", predicted_price)
        
        return(predicted_price)
        
        
    