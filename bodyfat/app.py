# from enum import Enum
 
# import joblib
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from xgboost import XGBRegressor
 
# from bodyfat.config import Config
# from bodyfat.features import create_new_features
 
 
# class Sex(str, Enum):
#     male = "M"
#     female = "F"
 
 
# preprocessor = joblib.load(Config.Path.MODELS_DIR / "preprocessor.joblib")
 
# model = XGBRegressor()
# model.load_model(Config.Path.MODELS_DIR / "model.json")
 
# app = FastAPI(
#     title="Bodyfat Prediction API",
#     description="REST API to predict bodyfat percentage based on personal measurements",
# )
 
 
# class BodyfatPredictionRequest(BaseModel):
#     hip: float
#     abdomen: float
#     age: int
#     weight: float
#     height: float
#     sex: Sex
 
 
# class BodyfatPrediction(BaseModel):
#     bodyfat: float
 
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Bodyfat Prediction API"}

# @app.post("/predict", response_model=BodyfatPrediction)
# def make_prediction(input_data: BodyfatPredictionRequest):
#     input_df = pd.DataFrame([input_data.model_dump()])
#     input_df = create_new_features(input_df)
#     preprocessed_input = preprocessor.transform(input_df)
#     prediction = model.predict(preprocessed_input)[0]
#     return BodyfatPrediction(bodyfat=prediction)
 
 
from enum import Enum
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles  # Commented out
from pydantic import BaseModel
from xgboost import XGBRegressor

from bodyfat.config import Config
from bodyfat.features import create_new_features

class Sex(str, Enum):
    male = "M"
    female = "F"

preprocessor = joblib.load(Config.Path.MODELS_DIR / "preprocessor.joblib")

model = XGBRegressor()
model.load_model(Config.Path.MODELS_DIR / "model.json")

app = FastAPI(
    title="Bodyfat Prediction API",
    description="REST API to predict bodyfat percentage based on personal measurements",
)

class BodyfatPredictionRequest(BaseModel):
    hip: float
    abdomen: float
    age: int
    weight: float
    height: float
    sex: Sex

class BodyfatPrediction(BaseModel):
    bodyfat: float

@app.post("/predict", response_model=BodyfatPrediction)
def make_prediction(input_data: BodyfatPredictionRequest):
    input_df = pd.DataFrame([input_data.model_dump()])
    input_df = create_new_features(input_df)
    preprocessed_input = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_input)[0]
    return BodyfatPrediction(bodyfat=prediction)

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_content = Path("index.html").read_text()
    return HTMLResponse(content=html_content, status_code=200)

# Optional: Serve static files (if any)
# app.mount("/static", StaticFiles(directory="static"), name="static")  # Commented out
