import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Pipelines.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

from flask import Flask

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        df=data.get_data_as_df()
        logging.info("Data tranformed to dtaframe")
        preprocess=PredictPipeline()
        
        results=preprocess.predict(df)
        logging.info("prediction is done")
        return render_template("home.html",results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)  