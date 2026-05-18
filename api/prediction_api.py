import pandas as pd
import numpy as np

from fastapi import FastAPI

from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier

from src.ingestion.bigquery_loader import (
    fetch_data
)

from src.preprocessing.preprocess import (
    preprocess_pipeline
)


app = FastAPI(
    title="Distributed ML Prediction API"
)


print("\nLoading Training Dataset")


# Load training data
df = fetch_data(limit=5000)


# Preprocess dataset
df, scaler, corr, selected_features = (
    preprocess_pipeline(df)
)


# Features and target
X = df[selected_features]

y = df["risk_category"]


print("\nTraining API Prediction Model")


# Train model
model = RandomForestClassifier(
    random_state=42
)

model.fit(X, y)


print("\nPrediction API Ready")


# Input schema
class PredictionInput(BaseModel):

    confirmed: float

    log_confirmed: float

    previous_day_cases: float

    rolling_avg_cases: float

    growth_rate: float

    day: int

    month: int


# Root route
@app.get("/")
def home():

    return {

        "message":
            "Distributed ML API Running"

    }


# Prediction route
@app.post("/predict")
def predict(data: PredictionInput):

    input_df = pd.DataFrame([{

        "confirmed":
            data.confirmed,

        "log_confirmed":
            data.log_confirmed,

        "previous_day_cases":
            data.previous_day_cases,

        "rolling_avg_cases":
            data.rolling_avg_cases,

        "growth_rate":
            data.growth_rate,

        "day":
            data.day,

        "month":
            data.month

    }])

    # Predict
    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    return {

        "prediction":
            str(prediction),

        "probabilities":
            probabilities.tolist()

    }