import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pymongo import MongoClient
from sklearn.metrics import roc_curve, precision_recall_curve
from dotenv import load_dotenv
import os

# --- Configuración inicial ---
st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")
st.title("📊 Dashboard - Modelo Random Forest (Bank Marketing)")

# --- Conexión a MongoDB ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bank_marketing_db"]
metrics = list(db["results"].find({}, {"_id": 0}))

# --- Cargar modelo y dataset ---
model = joblib.load("model/model.pkl")
df = pd.read_csv("data/bank.csv", sep=";")

# --- Mostrar métricas ---
st.header("📈 Métricas del Modelo")
if metrics:
    m = metrics[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['Accuracy']:.2f}")
    col2.metric("Precision", f"{m['Precision']:.2f}")
    col3.metric("Recall", f"{m['Recall']:.2f}")
    col4.metric("F1 Score", f"{m['F1_Score']:.2f}")
    col5.metric("ROC AUC", f"{m['ROC_AUC']:.2f}")
else:
    st.warning("No se encontraron métricas en la base de datos.")

# --- Vista previa del dataset ---
st.header("📂 Vista Previa del Dataset")
st.dataframe(df.head())

# --- Distribución de variables ---
st.header("📊 Distribución de Variables")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución por trabajo")
    job_counts = df["job"].value_counts()
    st.bar_chart(job_counts)

with col2:
    st.subheader("Distribución por estado civil")
    marital_counts = df["marital"].value_counts()
    st.bar_chart(marital_counts)

# --- Matriz de Confusión ---
st.header("🧩 Matriz de Confusión")
if metrics:
    conf_matrix = m["Confusion_Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)

# --- Curvas ROC y Precision-Recall ---
st.header("📉 Curvas de Rendimiento del Modelo")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preprocesamiento para curvas
df_copy = df.copy()
df_copy["y"] = df_copy["y"].map({"yes": 1, "no": 0})
cat_cols = df_copy.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df_copy[col] = le.fit_transform(df_copy[col])

X = df_copy.drop("y", axis=1)
y = df_copy["y"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_proba = model.predict_proba(X_test)[:, 1]

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr, label="ROC Curve")
ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("Curva ROC")
st.pyplot(fig1)

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(recall, precision, label="Precision-Recall Curve")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Curva Precision-Recall")
st.pyplot(fig2)
