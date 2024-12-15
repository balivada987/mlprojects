import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from src.exceptions import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
import os
from src.utils import save_object
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numeric_columns = ["reading_score", "writing_score"]
            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot_encode", OneHotEncoder())
            ])

            logging.info("Reading of numeric and categorical pipelines done")

            # Correct ColumnTransformer initialization
            preprocessing = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns)
                ]
            )

            logging.info("Transformation on two columns is done")
            
            return preprocessing

        except CustomException as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("train test file reading done in initiate_data_transformation")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "math_score"
            numerical_column_name = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            print(input_feature_train_df)
            logging.info("target Column Dropped in train and test")
            
            scaled_train = preprocessing_obj.fit_transform(input_feature_train_df)
            scaled_test = preprocessing_obj.transform(input_feature_test_df)  # Use transform instead of fit_transform for test data
            print(scaled_train)
            print(scaled_test)
            logging.info("train and test executed")

            train_df = np.c_[scaled_train, np.array(train_df[target_column_name])]
            test_df = np.c_[scaled_test, np.array(test_df[target_column_name])]
            logging.info("merged train, test with target")

            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path, obj=preprocessing_obj)
            logging.info("saved pipeline dill file")

            return train_df, test_df, self.data_transformation_config.preprocess_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
