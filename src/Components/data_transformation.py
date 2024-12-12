
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import SimpleImputer,StandardScaler,OneHotEncoder
from sklearn.preprocessing import Pipeline
from src.exceptions import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join('artifacts', 'preprocess.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            numeric_columns=[]
            cat_columns=[]
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                 ("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())])

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # "mode" is not valid, use "most_frequent"
                    ("onehot_encode", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            ) 

        
                       
        except CustomException as e:
            pass


        

            
