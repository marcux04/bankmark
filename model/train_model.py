import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# --- Cargar variables de entorno ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# --- Conectar a MongoDB ---
client = MongoClient(MONGO_URI)
db = client["bank_marketing_db"]
results_collection = db["results"]

# --- 1. Cargar el dataset ---
print("📂 Cargando dataset...")
df = pd.read_csv("data/bank.csv", sep=";")

# --- 2. Preprocesamiento ---
print("⚙️ Preprocesando datos...")

# Convertir variable objetivo
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Detectar variables categóricas
cat_cols = df.select_dtypes(include=["object"]).columns

# Codificación con LabelEncoder (más sencillo para Random Forest)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# --- 3. Separar variables predictoras y objetivo ---
X = df.drop("y", axis=1)
y = df["y"]

# --- 4. Dividir dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 5. Entrenar modelo ---
print("🌲 Entrenando modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 6. Predicciones ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- 7. Calcular métricas ---
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1_Score": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, y_prob),
    "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
}

# --- 8. Guardar modelo ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("✅ Modelo guardado como 'model/model.pkl'")

# --- 9. Guardar métricas en MongoDB ---
results_collection.insert_one(metrics)
print("📊 Métricas guardadas en MongoDB con éxito")

# --- 10. Mostrar resultados ---
print("\n🔎 RESULTADOS DEL MODELO:")
for key, value in metrics.items():
    if key != "Confusion_Matrix":
        print(f"{key}: {value:.4f}")
