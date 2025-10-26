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
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# --- Conectar a MongoDB ---
try:
    client = MongoClient(MONGO_URI)
    db = client["bank_marketing_db"]
    results_collection = db["results"]
    print("✅ Conexión a MongoDB establecida.")
except Exception as e:
    print(f"❌ Error al conectar a MongoDB: {e}")
    exit(1)

# --- 1. Cargar el dataset ---
DATA_PATH = "data/bank.csv"
print("📂 Cargando dataset...")
try:
    df = pd.read_csv(DATA_PATH, sep=";")
except FileNotFoundError:
    print(f"❌ No se encontró el archivo {DATA_PATH}.")
    exit(1)
except Exception as e:
    print(f"❌ Error al cargar el dataset: {e}")
    exit(1)

# --- 2. Preprocesamiento ---
print("⚙️ Preprocesando datos...")
# Convertir variable objetivo
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Detectar variables categóricas
cat_cols = df.select_dtypes(include=["object"]).columns

# Codificación con LabelEncoder y guardar los codificadores
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

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

# --- 8. Guardar modelo y codificadores ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
print("✅ Modelo y codificadores guardados como 'model/model.pkl' y 'model/label_encoders.pkl'")

# --- 9. Guardar métricas en MongoDB ---
try:
    results_collection.delete_many({})  # Limpiar métricas previas
    results_collection.insert_one(metrics)
    print("📊 Métricas guardadas en MongoDB con éxito")
except Exception as e:
    print(f"❌ Error al guardar métricas en MongoDB: {e}")

# --- 10. Mostrar resultados ---
print("\n🔎 RESULTADOS DEL MODELO:")
for key, value in metrics.items():
    if key == "Confusion_Matrix":
        print(f"{key}: {value}")
    elif isinstance(value, (int, float)):  # Solo formatear como float si es numérico
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")  # Imprimir sin formato si no es numérico