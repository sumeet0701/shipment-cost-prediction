# Supply Chain Shipment Pricing Prediction
##  Problem Statements:

The market for supply chain analytics is expected to develop at a CAGR of 17.3 percent 
from 2019 to 2024, more than doubling in size. This data demonstrates how supply 
chain organizations are understanding the advantages of being able to predict what will 
happen in the future with a decent degree of certainty. Supply chain leaders may use 
this data to address supply chain difficulties, cut costs, and enhance service levels all at 
the same time.

## Goal of Project:
The main goal is to predict the supply chain shipment pricing based on the available 
factors in the dataset.

## Proposed Solution Approach:
1. First of all, Exploratory Data Analysis (EDA), Feature Engineering (FE) and Feature Selection (FS) [if required] using various python based libraries [pandas, numpy etc.] on downloaded data set from the above mentioned link will be performed. 
2. Visualization tools [matplotlib, seaborn etc.] will aid to get a better understanding of the data that we are working with. Afterwards, distinct regression models wiil be created. 
3. Finally, We will evaluate these models using distinct perfomance metrics plus will try to get best Hyper prameters using Grid Search CV apporach and will select the best performing(most suitable) model for this specific dataset for predictions of heating load as well as cooling load of residential buildings."

## dataset:
Kaggle: https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data


## 🛠 Skills
●	Python Programming
    ○	OOPS Concept
    ○	Modularity

●	Library used

    ○	 Numpy
    ○	Pandas
    ○	Matplotlib
    ○	Seaborn
    ○	Sklearn
    ○	evidently (Data Drift)
    ○	optuna
    ○	Pymongo

●	Machine Learning Algorithms
    ○	XGBoost Regression
    ○	Random Forest Regressor
    ○	Decision Tree Regressor

●	Machine Learning

    ○	Single Prediction
    ○	Batch Prediction

●	AWS Cloud

●	Mlops ( DVC & Mlflow for tracking )

●	MongoDB Database

●	Docker

●	Flask API

●	Version: Git 

●	HTML, CSS, JS for Designing


## Project Details
There are six packages in the pipeline: Config, Entity, Constant, Exception, Logger, Components and Pipeline

### Config
This package will create all folder structures and provide inputs to the each of the components.

### Entity
This package will defines named tuple for each of the components config and artifacts it generates.

### Constant
This package will contain all predefined constants which can be used accessed from anywhere

### Exception
This package contains the custom exception class for the Prediction Appliaction

### Logger
This package helps in logging all the activity

# Components
--------
## Data Ingestion 
-----
### Folder structure 

![Before](https://user-images.githubusercontent.com/109200332/226115648-39a3c045-c68f-4a44-8398-2d643aa9fec9.png)


#### Data Ingestion 
This module downloads the data from the link, unzip it, then stores entire data into Db.
From DB it extracts all data into single csv file and split it into training and testing datasets.

![Data_Ingestion](https://user-images.githubusercontent.com/109200332/226117526-e5669825-d7e4-4e9a-8347-8ce11d314386.png)


## Data Validation

Data Validation: This module validates whether data files passed are as per defined schema which was agreed upon by client.


![Data_Validation](https://user-images.githubusercontent.com/109200332/226121268-9ef2e4ca-21d1-4f9b-a6f5-cd8c15323bc4.png)


## Data Transformation

This module applies all the Feature Engineering and preprocessing to the data we need to 
train our model and save  the pickle object for same.

![Data Transformation](https://user-images.githubusercontent.com/109200332/226129709-116764b4-8eab-43e8-bacb-934ad7f2ad2a.png)

## Model Trainer
 This module trains the model on transformed data, evalutes it based on R2 accuracy score and 
 saves the best performing model object for prediction

![Model Trainer](https://user-images.githubusercontent.com/109200332/226136355-3704614b-c6e6-4eb7-b39c-e29ce9127847.png)

### Pipeline
This package contains two modules:
1. Training Pipeline: This module will initiate the training pipeline where each of the above mentioned components  
                      will be called sequentially untill model is saved.
2. Prediction Pipeline: This module will help getting prediction from saved trained model.

## Web Application Buliding:
![web Application](![App](https://github.com/sumeet0701/shipment-cost-prediction/assets/63961794/4ba8f4cc-3eb4-4139-afc4-e0e5a72f85f4)


## Project Archietecture
![Supply Chain prediction](https://github.com/sumeet0701/shipment-cost-prediction/assets/63961794/4250b8d5-eab3-4351-83a4-6b16a1b5b899)


## Development Archietecture
![deploment](https://github.com/sumeet0701/shipment-cost-prediction/assets/63961794/5003354d-4024-4fc8-b21c-a5bd77acfe2f)


## Installation

Install my-project with npm

### Step 1 - Clone the repository
```bash
git clone https://github.com/sumeet0701/shipment-cost-prediction.git
```

### Step 2 - Create a conda environment after opening the repository

```bash
conda create -n env_name python
```

```bash
conda activate env_name
```

### Step 3 - Install the requirements
```bash
pip3 install -r requirements.txt
```


### Step 4 - Train Model
```bash
python demo.py

```

### Step g - Run the Prediction application server [Using Flask API]
```bash
python main.py
```

### Step 7 - Prediction application [Using Flask API]
```bash
http://localhost:5000

```







## 🔗 Links
[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/sumeet0701/)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sumeet-maheshwari/)





## Authors

- [@Maheshwari Sumeet](https://github.com/sumeet0701)
- [@ Hitesh Nimbalkar](https://github.com/Hitesh-Nimbalkar)

