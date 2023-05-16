import os
import logging
from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 



BATCH_PREDICTION = "batch_prediction"
INSTANCE_PREDICTION="Instance_prediction"
input_file_path="dataset.csv"
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
BRAND_MAP = {'Generic': 0, 'Others': 1, 'Determine': 2, 'Uni-Gold': 3}
FIRST_LINE_DESIGNATION_MAP = {'Yes': 0, 'No': 1}



class instance_prediction_class:
    def __init__(self) -> None:
        pass
    
    def preprocess_input(self,pack_price, unit_price, weight_kg, line_item_quantity, fulfill_via, shipment_mode, country, brand, sub_classification, first_line_designation):
        # Convert categorical variables to numerical format
        fulfill_via = FULFILL_VIA_MAP[fulfill_via]
        shipment_mode = SHIPMENT_MODE_MAP[shipment_mode]
        country = COUNTRY_MAP[country]
        brand = BRAND_MAP[brand]
        sub_classification = SUB_CLASSIFICATION_MAP[sub_classification]
        first_line_designation = FIRST_LINE_DESIGNATION_MAP[first_line_designation]

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'Pack_Price': [pack_price],
            'Unit_Price': [unit_price],
            'Weight_Kilograms_Clean': [weight_kg],
            'Line_Item_Quantity': [line_item_quantity],
            'Fulfill_Via': [fulfill_via],
            'Shipment_Mode': [shipment_mode],
            'Country': [country],
            'Brand': [brand],
            'Sub_Classification': [sub_classification],
            'First_Line_Designation': [first_line_designation]
        })

        # Preprocess the user input using the preprocessor
        preprocessed_input = preprocessor.transform(user_input)

        # Return the preprocessed input as a numpy array
        return preprocessed_input.toarray()

    def predict_price(preprocessed_input):
        # Make a prediction using the pre-trained model
        predicted_price = model.predict(preprocessed_input)

        # Return the predicted shipment price
        return predicted_price[0]

    def predict_price_from_input(self):
        # Get input from the user
        pack_price = float(input("Enter the pack price: "))
        unit_price = float(input("Enter the unit price: "))
        weight_kg = float(input("Enter the weight in kilograms: "))
        line_item_quantity = int(input("Enter the line item quantity: "))
        fulfill_via = input("Enter the fulfill via (From RDC/Direct Drop): ")
        shipment_mode = input("Enter the shipment mode (Truck/Air/Air Charter/Ocean): ")
        country = input("Enter the country (Zambia/Ethiopia/Nigeria/Tanzania/Côte d'Ivoire/Mozambique/Others/Zimbabwe/South Africa/Rwanda/Haiti/Vietnam/Uganda): ")
        brand = input("Enter the brand (Generic/Others/Determine/Uni-Gold): ")
        sub_classification = input("Enter the sub-classification (Adult/Pediatric/HIV test/HIV test - Ancillary/Malaria/ACT): ")
        first_line_designation = input("Enter the first line designation (Yes/No): ")

        # Preprocess the input using the preprocessor
        preprocessed_input = self.preprocess_input(pack_price, unit_price, weight_kg, line_item_quantity, fulfill_via, shipment_mode, country, brand, sub_classification, first_line_designation)

        # Make a prediction using the pre-trained model
        predicted_price = self.predict_price(preprocessed_input)

        # Print the predicted shipment price
        print("The predicted shipment price is: $", predicted_price)
        
        return(predicted_price)
        
        
    