from shipment_cost_prediction.logger import logging
from shipment_cost_prediction.exception import CustomException
from shipment_cost_prediction.entity.config_entity import DataTransformationConfig
from shipment_cost_prediction.entity.artifact_entity import *
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.utils.utils import read_yaml_file
from shipment_cost_prediction.utils.utils import save_data
from shipment_cost_prediction.utils.utils import save_object


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from numpy import asarray
import os,sys
import pandas as pd
import numpy as np
import re



class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,numerical_columns,categorical_columns,target_columns,drop_columns):
        """
        This class applies necessary Feature Engneering for Shipment cost prediction Data
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_columns = target_columns
        self.columns_to_drop = drop_columns
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engoneering Pipeline ")
        
    def Map_encoding(self,x):
        try:
            Encode_Features=[]
            # Check for non-float columns in x and include them in the encoding feature list
            for col in x.columns:
                if x[col].dtype == "object" or x[col].dtype == "category":
                    if col not in Encode_Features:
                        Encode_Features.append(col)
                        
            
            # x.to_csv('Before_Encoding.csv', index=False)
            logging.info(f"Columns before encoding: {x.columns}")
            
            # Print information about each feature to be encoded
            for feature in Encode_Features:
                unique_values = x[feature].unique()
                logging.info(f"Unique values of {feature}: {unique_values}")
                logging.info("\n")
                logging.info(f"Number of unique values of {feature}: {len(unique_values)}")

            # Define the mapping for each feature
            mapping = {}
            for feature in Encode_Features:
                unique_values = x[feature].unique()
                mapping[feature] = {value: idx for idx, value in enumerate(unique_values)}
                logging.info(f"Mapping for {feature}: {mapping[feature]}")
                
            # Use the mapping to encode each feature and add encoded values to dataframe
            for feature in Encode_Features:
                encoded_values = x[feature].map(mapping[feature])
                x[f"{feature}"] = encoded_values  
            
            return x

    
        except Exception as e:
            raise CustomException(e,sys) from e 
            
      
    def data_modification(self,x):
        try:
            
            if 'Manufacturing_Site' in x.columns:
                # Manufacturing Site - "Others"
                counts = x['Manufacturing_Site'].value_counts()
                idx = counts[counts.lt(20)].index
                x.loc[x['Manufacturing_Site'].isin(idx), 'Manufacturing_Site'] = 'Others'

            if 'Country' in x.columns:
                # Country 
                counts = x['Country'].value_counts()
                idx = counts[counts.lt(30)].index
                x.loc[x['Country'].isin(idx), 'Country'] = 'Others'

            if 'Brand' in x.columns:
                # Brand 
                counts = x['Brand'].value_counts()
                idx = counts[counts.lt(50)].index
                x.loc[x['Brand'].isin(idx), 'Brand'] = 'Others'

            if 'Dosage_Form' in x.columns:
                # Dosage Form modification
                x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Tablet",case=False)),"Tablet",x["Dosage_Form"])
                x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Oral",case=False)),"Oral",x["Dosage_Form"])
                x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("Capsule",case=False)),"Capsule",x["Dosage_Form"])
                x["Dosage_Form"]=np.where((x['Dosage_Form'].str.contains("kit",case=False)),"Test kit",x["Dosage_Form"])
            
            logging.info(
                    f" >>>>>>>>>>>>  Columns Modififcation Complete  <<<<<<<<<<")
           
            return x
        except Exception as e:
            raise CustomException(e,sys) from e 
        

    def drop_columns(self,x):
        try:
            
            # define original column names of x
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

            # specify columns to drop initially
            columns_to_drop = self.drop_columns
            # drop the specified columns from x
            x.drop(columns=columns_to_drop, inplace=True)
            logging.info("Drop Columns Complete")
            
            return x
        except Exception as e:
            raise CustomException(e,sys) from e 
        
            
    def Missing_fills(self,x):
        try:
            # Shipment Mode - Mode 
            if 'Shipment_Mode' in x.columns:
                mode=x['Shipment_Mode'].mode()
                x['Shipment_Mode']=x['Shipment_Mode'].fillna(mode[0])
                logging.info('Filled missing values in column "Shipment_Mode" with mode value:', mode[0])

            # Manufacturing Site - Mode 
            if 'Manufacturing_Site' in x.columns:
                mode=x['Manufacturing_Site'].mode()
                x['Manufacturing_Site']=x['Manufacturing_Site'].fillna(mode[0])
                logging.info('Filled missing values in column "Manufacturing_Site" with mode value:', mode[0])

            # Country - Mode 
            if 'Country' in x.columns:
                mode=x['Country'].mode()
                x['Country']=x['Country'].fillna(mode[0])
                logging.info('Filled missing values in column "Country" with mode value:', mode[0])

            # Brand 
            if 'Brand' in x.columns:
                mode=x['Brand'].mode()
                x['Brand']=x['Brand'].fillna(mode[0])
                logging.info('Filled missing values in column "Brand" with mode value:', mode[0])

            return x
        except Exception as e:
            raise CustomException(e,sys) from e 
        

    def outlier(self,x):
        try:
            logging.info("Outlier Detection")
            # Select all float columns except Freight_Cost_USD_Clean
            float_cols = [col for col in x.columns if x[col].dtype == 'float' and col != 'Freight_Cost_USD_Clean']
                
            for i in float_cols:
                Q1 = np.percentile(x[i], 25)
                Q3 = np.percentile(x[i], 75)
                IQR = Q3 - Q1
                lower_limit = Q1 - (1.5 * IQR)
                upper_limit = Q3 + (1.5 * IQR)   
                
                print('Column:', i)
                print('Q1:', Q1)
                print('Q3:', Q3)
                print('IQR:', IQR)
                print('Upper limit:', upper_limit)
                print('Lower limit:', lower_limit)
                
                median = x[i].median()
                x[i] = np.where(x[i]>upper_limit, median, np.where(x[i]<lower_limit, median, x[i]))
                print("\n")
                logging.info('No. of outliers detected:', len(np.where((x[i] > upper_limit) | (x[i] < lower_limit))[0]))
                
            
            logging.info("Outlier Detection Complete")

            return x
        except Exception as e:
                raise CustomException(e,sys) from e 
    
        
    def Weight_and_freight(self,x):
        try:
            
            regex = {"id_number": ":\d*"
                                    }
            def change_to_number(freight_cost_usd):
                regex = {
                                        "id_number": ":\d*"
                                    }
                match = re.search(regex['id_number'], freight_cost_usd, re.IGNORECASE)
                if match:
                    id = match.group(0).replace(':','')
                    filtered = x.query("ID == "+id)
                    if not filtered.empty:
                        return filtered['Freight_Cost_(USD)'].iloc[0]
                return freight_cost_usd
            


            def convert_to_number(weight):
                regex = {
                                        "id_number": ":\d*"
                                    }
                match = re.search(regex['id_number'], weight, re.IGNORECASE)
                if match:
                    id = match.group(0).replace(':','')
                    filtered = x.query("ID == "+id)
                    if not filtered.empty:
                        return filtered['Weight_(Kilograms)'].iloc[0]
                return weight   


                
                
            x['Freight_Cost_USD_Clean'] = x['Freight_Cost_(USD)'].apply(change_to_number)
            
            x['Weight_Kilograms_Clean'] = x['Weight_(Kilograms)'].apply(convert_to_number)
            
            print("Weight and freight completed")
                        
            freight_cost_indexes = x.index[(x['Freight_Cost_USD_Clean'] == 'Freight Included in Commodity Cost') | (x['Freight_Cost_USD_Clean'] == 'Invoiced Separately')].tolist()
            weight_indexes = x.index[x['Weight_Kilograms_Clean'] == 'Weight Captured Separately'].tolist()
            shipment_indexes = x.index[x['Shipment_Mode'] == 'no_value'].tolist()
            print("Freight_Cost_USD_Clean_indexes:",len(freight_cost_indexes))
            print("Weight_Kilograms_Clean_indexes:",len(weight_indexes))
            print("Shipment_Mode indexes:         ",len(shipment_indexes))

            indexes = list(set(freight_cost_indexes + weight_indexes + shipment_indexes))
            print("Indexes:",len(indexes))
            df_clean = x.drop(indexes)
            
            df_clean = df_clean[~df_clean['Freight_Cost_USD_Clean'].str.contains('See')]
            df_clean = df_clean[~df_clean['Weight_Kilograms_Clean'].str.contains('See')]
            

            
            df_clean["Freight_Cost_USD_Clean"]=df_clean["Freight_Cost_USD_Clean"].astype("float")
            df_clean["Weight_Kilograms_Clean"]=df_clean["Weight_Kilograms_Clean"].astype("float")
            
            
            print(df_clean.shape)
            
            logging.info('Weight and Freight Cost Data Clean Completed')
        
            
            return df_clean
            
        except Exception as e:
            raise CustomException(e,sys) from e
        

    
    def data_wrangling(self,x):
        try:
            # Weight_(kilograms) and Freight_Cost Data Clean 
            
            data=self.Weight_and_freight(x)
            
            # Dropping Columns 
            data = self.drop_columns(data)
            
            # Filling Missing Data 
            data= self.Missing_fills(data)
            
            
            # Data Modification 
            data = self.data_modification(data)
            
            # Outlier Detection 
            
            data = self.outlier(data)
            
            
            

            # Perform map encoding
            data = self.Map_encoding(data)
           

            #data.to_csv("data_modiefied.csv",index=False)
            logging.info('Data Modified  Completed and Saved ')
            
            return data
    
        
        except Exception as e:
            raise CustomException(e,sys) from e
            
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        try:
            X = self.data_wrangling(X)
            numerical_columns = self.numerical_columns
            categorical_columns=self.categorical_columns
            target_column=self.target_columns
            
            
            col = numerical_columns+categorical_columns+target_column
            print("\n")
            logging.info(f"New Column Order {col}")
            print("\n")
            X = X[col]
            X.to_csv('Data_Transform Complete.csv', index=False)
            arr = X.values
            
            return arr
        except Exception as e:
            raise CustomException(e,sys) from e






class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

             ## Accesssing Column Labels 
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            self.schema = read_yaml_file(file_path=self.schema_file_path)
            self.target_column_name = self.schema[TARGET_COLUMN_KEY]
            self.numerical_columns = self.schema[NUMERICAL_COLUMN_KEY] 
            self.categorical_columns = self.schema[CATEGORICAL_COLUMN_KEY]
            self.drop_columns=self.schema[DROP_COLUMN_KEY]

        except Exception as e:
            raise CustomException(e,sys) from e
        
        
    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                target_columns=self.target_column_name,
                drop_columns=self.drop_columns))])
            
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def get_data_transformer_object(self):
        try:
            logging.info('Creating Data Transformer Object')

            numerical_columns = self.numerical_columns
            categorical_columns =self.categorical_columns
            # Define transformers for numerical and categorical columns
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])

            # Combine the transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, numerical_columns),
                    ('cat', cat_transformer, categorical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) from e
        
        
        


    def initiate_data_transformation(self):
        try:
            logging.info(f"Obtaining training and test file path.")
            train_file_path =  self.data_validation_artifact.validated_train_path
            test_file_path = self.data_validation_artifact.validated_test_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            # Reading schema file for columns details
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_file_path)
            
            logging.info(f"Extracting train column name {train_df.columns.to_list()}")

            # Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns = self.numerical_columns
            categorical_columns = self.categorical_columns
            
            # Log column information
            logging.info(f"Numerical columns {numerical_columns}")
            logging.info(f"Categorical columns {categorical_columns}")
            logging.info(f"Target Column :{target_column_name}")
            
            col = numerical_columns+categorical_columns+target_column_name
            logging.info(f"All columns : {col}")
            print(col)


            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            
            
            logging.info(f"Feature Enineering - Train Data ")
            feature_eng_train_arr = fe_obj.fit_transform(train_df)
            
            logging.info(f"Feature Enineering - Test Data ")
            feature_eng_test_arr = fe_obj.transform(test_df)
            
            # Converting featured engineered array into dataframe
            logging.info(f"Converting featured engineered array into dataframe.")            
            logging.info(f"Columns for Feature Engineering : {col}")
            feature_eng_train_df = pd.DataFrame(feature_eng_train_arr,columns=col)
            logging.info(f"Feature Engineering - Train Completed")
            feature_eng_test_df = pd.DataFrame(feature_eng_test_arr,columns=col)
            
            logging.info(f"Saving feature engineered training and testing dataframe.")
            #feature_eng_train_df.to_csv('feature_eng_train_df.csv',index=False)

           
            # Tran and TEst Dataframe
            target_column_name='Freight_Cost_USD_Clean'

            target_feature_train_df = feature_eng_train_df[target_column_name]
            input_feature_train_df = feature_eng_train_df.drop(columns = target_column_name,axis = 1)
             
            target_feature_test_df = feature_eng_test_df[target_column_name]
            input_feature_test_df = feature_eng_test_df.drop(columns = target_column_name,axis = 1)
            
            #input_feature_train_df.to_csv('input_feature_train_df.csv',index=False)
  
            
            
            ## Preprocessing 

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            preprocessing_obj = self.get_data_transformer_object()
           
            
            col =numerical_columns+categorical_columns

            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)

            transformed_train_df = pd.DataFrame(np.c_[train_arr,np.array(target_feature_train_df)],columns=col+[target_column_name])
            transformed_test_df = pd.DataFrame(np.c_[test_arr,np.array(target_feature_test_df)],columns=col+[target_column_name])

            
        
            # Adding target column to transformed dataframe
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir    
        
            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")
            transformed_test_file_path = os.path.join(transformed_test_dir,"transformed_test.csv")
            

            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test file")
            
            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            save_data(file_path = transformed_test_file_path, data = transformed_test_df)
            logging.info("Transformed Train and Transformed test file saved")


            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path = transformed_train_file_path,
            transformed_test_file_path = transformed_test_file_path,
            preprocessed_object_file_path = preprocessing_object_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")