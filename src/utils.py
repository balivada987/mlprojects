import pandas as pd
import numpy as np
import os
import sys
import dill
from src.logger import logging

from src.exceptions import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            logging.info(" pickle file dumped")
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path: str):
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test, models, params:dict):
    try:
        report= {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=list(params.values())[i]
            X_train=X_train
            y_train=y_train
            #rs=RandomizedSearchCV(model,param)
            model.fit(X_train,y_train) 
            #model.set_params(**rs.best_params_) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # Evaluate Train and Test dataset
            model_train_r2 = r2_score(y_test[:199],y_test_pred[:199])
            report[list(models.keys())[i]]= model_train_r2
        return report
    except Exception as e:
        raise CustomException(e, sys)
    


