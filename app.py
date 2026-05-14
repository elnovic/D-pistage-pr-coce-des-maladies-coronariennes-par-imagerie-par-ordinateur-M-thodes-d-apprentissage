from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles avec chemin absolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'heart_disease_rf_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class PatientData(BaseModel):
    age: float
    sex: int
    chest_pain_type: int
    resting_bp: float
    cholesterol: float
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    st_slope: int

@app.get("/")
def read_root():
    return {"message": "Heart Disease Detection API", "status": "running"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array([[
            data.age,
            data.sex,
            data.chest_pain_type,
            data.resting_bp,
            data.cholesterol,
            data.fasting_blood_sugar,
            data.resting_ecg,
            data.max_heart_rate,
            data.exercise_angina,
            data.oldpeak,
            data.st_slope
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        return {
            "prediction": int(prediction),
            "probability_normal": float(probability[0]),
            "probability_disease": float(probability[1]),
            "risk_level": "Élevé" if probability[1] > 0.7 else "Modéré" if probability[1] > 0.3 else "Faible"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Handler pour Vercel (ASGI → serverless)
handler = Mangum(app)
