# Put the code for your API here.
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel, Field
import uvicorn
from joblib import load
import os.path
import sys

# load model
model = load("starter/model_building/trainedmodel.pkl")

# instantiate app with fastapi
app = FastAPI()

# allow heroku to pull data from dvc
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
# greeting
@app.get("/")
async def greet_user():
    return {"Welcome!"}


# greet with name
@app.get("/{name}")
async def get_name(name: str):
    return {f"Hi {name}, Welcome to this app"}


# models with pydantic
class ClassifierFeatureIn(BaseModel):
    age: int = Field(..., example=50)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2500, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


@app.post("/predict")
def predict(data1: ClassifierFeatureIn):
    df = pd.read_csv('../data/census_cleaned.csv')
    # predict
    pred, y = inference(df)
    return {
        "prediction": preds[0]
    }

"""
# pydantic output of the model
class ClassifierOut(BaseModel):
    # The forecast output will be either >50K or <50K
    forecast: str = "Income <=50k"
"""    
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
