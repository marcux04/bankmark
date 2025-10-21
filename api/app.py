from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# --- Cargar entorno y modelo ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bank_marketing_db"]
results_collection = db["results"]

# --- Cargar modelo entrenado ---
model = joblib.load("model/model.pkl")

# --- Crear aplicación FastAPI ---
app = FastAPI(title="Bank Marketing API", version="1.0")

# --- Definir esquema de entrada ---
class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# --- Endpoint de bienvenida ---
@app.get("/")
def home():
    return {"message": "API de Random Forest - Bank Marketing"}

# --- Endpoint para ver métricas ---
@app.get("/metrics")
def get_metrics():
    metrics = list(results_collection.find({}, {"_id": 0}))
    return {"metrics": metrics}

# --- Endpoint para predicciones ---
@app.post("/predict")
def predict(client: ClientData):
    data = client.dict()
    df = pd.DataFrame([data])

    # Codificar variables categóricas igual que en el entrenamiento
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_yes": round(float(proba), 3),
        "message": "El cliente probablemente contrate el servicio" if prediction == 1 else "El cliente NO contratará el servicio"
    }
