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
    title="Big Data ML API"
)


print("\nLoading Training Dataset")


# Load historical data
df = fetch_data(limit=5000)


# Preprocess
df, scaler, corr, selected_features = (
    preprocess_pipeline(df)
)


print("\nTraining API ML Model")


# Features
X = df[selected_features]

# Labels
y = df["risk_category"]


# Train lightweight API model
model = RandomForestClassifier(
    random_state=42
)

model.fit(X, y)


print("\nAPI Model Ready")


# Input schema
class PredictionInput(BaseModel):

    confirmed: float

    log_confirmed: float

    previous_day_cases: float

    rolling_avg_cases: float

    growth_rate: float

    day: int

    month: int


@app.get("/")
def home():

    return {

        "message":
            "Big Data ML API Running"

    }


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

    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    alert = "NORMAL"

    if prediction == "High":

        alert = "HIGH ALERT"

    elif prediction == "Medium":

        alert = "WARNING"


    return {

        "prediction":
            prediction,

        "confidence":
            round(confidence, 4),

        "alert":
            alert

    }