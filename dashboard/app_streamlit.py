import streamlit as st
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import os
from sklearn.preprocessing import LabelEncoder

# --- Función para cargar CSV con detección automática de separador ---
def load_csv_smart(file_path_or_buffer):
    try:
        # Detectar automáticamente el separador (coma o punto y coma)
        sample = pd.read_csv(file_path_or_buffer, sep=None, engine="python", nrows=5)
        return pd.read_csv(file_path_or_buffer, sep=sample.attrs["delimiter"], engine="python")
    except Exception:
        # Si falla la detección automática, probar manualmente
        try:
            return pd.read_csv(file_path_or_buffer, sep=";")
        except:
            return pd.read_csv(file_path_or_buffer, sep=",")

# --- Función para preprocesar datos de entrada ---
def preprocess_input(input_data, label_encoders, feature_columns):
    df_input = pd.DataFrame([input_data])
    for col in label_encoders:
        if col in df_input.columns:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col])
            except ValueError:
                st.error(f"❌ Valor inválido para {col}. Asegúrate de usar valores válidos.")
                return None
    # Asegurar que las columnas estén en el mismo orden que en el entrenamiento
    df_input = df_input[feature_columns]
    return df_input

# 🔧 Configuración general
st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")
st.title("📊 Dashboard - Bank Marketing (Random Forest)")

# 🌍 Conexión con MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI)
    db = client["bank_marketing_db"]
    st.info("✅ Conexión a MongoDB establecida.")
except Exception as e:
    st.error(f"❌ Error al conectar a MongoDB: {e}")
    st.stop()

# 📂 Cargar modelo y codificadores
try:
    model = joblib.load("model/model.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")
    st.info("✅ Modelo y codificadores cargados correctamente.")
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos 'model.pkl' o 'label_encoders.pkl'.")
    st.stop()

# Obtener las columnas del modelo (para asegurar el orden correcto)
feature_columns = model.feature_names_in_.tolist()

# 🌍 Configuración de la API (opcional)
API_URL = "http://127.0.0.1:8000/predict"

st.sidebar.header("Opciones")

# ==========================================================
# 🧾 SECCIÓN 1: Cargar dataset (predeterminado o del usuario)
# ==========================================================
st.subheader("🆕 Cargar dataset (CSV)")
DATA_PATH = "data/bank.csv"
data = None
uploaded_file = st.file_uploader("Selecciona un archivo CSV para actualizar datos", type=["csv"])

if uploaded_file is not None:
    # 📥 Si el usuario sube un CSV, se usa ese
    data = load_csv_smart(uploaded_file)
    st.info("✅ Se ha cargado el archivo del usuario.")
else:
    # 📄 Si no sube nada, se usa el dataset por defecto
    if os.path.exists(DATA_PATH):
        data = load_csv_smart(DATA_PATH)
        st.info(f"📂 Cargando dataset predeterminado: {DATA_PATH}")
    else:
        st.error(f"❌ No se encontró el archivo predeterminado '{DATA_PATH}'.")
        st.stop()

# Mostrar vista previa
st.write("Vista previa del dataset actual:")
st.dataframe(data.head())

# Botón para actualizar MongoDB
if st.button("Actualizar MongoDB con este dataset"):
    try:
        data_dict = data.to_dict(orient="records")
        db["bank_clients"].delete_many({})  # Limpia la colección previa
        db["bank_clients"].insert_many(data_dict)
        st.success("✅ Base de datos actualizada correctamente con el dataset actual.")
    except Exception as e:
        st.error(f"❌ Error al actualizar MongoDB: {e}")

# ==========================================================
# 🔮 SECCIÓN 2: Predicción manual
# ==========================================================
st.subheader("🔮 Realizar una predicción manual")
with st.form("prediction_form"):
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    job = st.selectbox("Trabajo", ["admin.", "technician", "services", "management", "blue-collar", "retired", "student", "unemployed", "entrepreneur", "housemaid", "unknown"])
    marital = st.selectbox("Estado civil", ["married", "single", "divorced"])
    education = st.selectbox("Educación", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("¿Tiene crédito por defecto?", ["yes", "no"])
    balance = st.number_input("Balance", value=1000)
    housing = st.selectbox("¿Tiene crédito de vivienda?", ["yes", "no"])
    loan = st.selectbox("¿Tiene préstamo personal?", ["yes", "no"])
    contact = st.selectbox("Tipo de contacto", ["cellular", "telephone", "unknown"])
    day = st.slider("Día del mes", 1, 31, 15)
    month = st.selectbox("Mes", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Duración (segundos)", value=300)
    campaign = st.number_input("Número de contactos en la campaña", value=1)
    pdays = st.number_input("Días desde el contacto previo", value=-1)
    previous = st.number_input("Número de contactos previos", value=0)
    poutcome = st.selectbox("Resultado previo", ["success", "failure", "unknown", "other"])
    
    input_data = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
    }
    
    submitted = st.form_submit_button("Predecir")
    if submitted:
        # Preprocesar los datos de entrada
        processed_input = preprocess_input(input_data, label_encoders, feature_columns)
        if processed_input is not None:
            try:
                # Intentar usar la API
                response = requests.post(API_URL, json=input_data, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Predicción completada (vía API): {result['message']}")
                    st.write(f"**Probabilidad de contratar el servicio:** {result['probability_yes']:.2f}")
                else:
                    st.warning("⚠️ API no disponible, usando modelo local.")
                    # Usar el modelo local como respaldo
                    prediction = model.predict(processed_input)[0]
                    probability = model.predict_proba(processed_input)[0][1]
                    message = "El cliente probablemente SÍ contratará el servicio." if prediction == 1 else "El cliente probablemente NO contratará el servicio."
                    st.success(f"✅ Predicción completada (local): {message}")
                    st.write(f"**Probabilidad de contratar el servicio:** {probability:.2f}")
            except requests.exceptions.RequestException:
                # Usar el modelo local si la API falla
                prediction = model.predict(processed_input)[0]
                probability = model.predict_proba(processed_input)[0][1]
                message = "El cliente probablemente SÍ contratará el servicio." if prediction == 1 else "El cliente probablemente NO contratará el servicio."
                st.success(f"✅ Predicción completada (local): {message}")
                st.write(f"**Probabilidad de contratar el servicio:** {probability:.2f}")

# ==========================================================
# 📈 SECCIÓN 3: Métricas y visualizaciones
# ==========================================================
st.subheader("📈 Métricas y Gráficos del Modelo")
try:
    metrics = db["results"].find_one({}, {"_id": 0})
    if metrics:
        st.json(metrics)
    else:
        st.warning("⚠️ No se encontraron métricas guardadas en la colección 'results'.")
except Exception as e:
    st.error(f"❌ Error al leer métricas de MongoDB: {e}")

# Mostrar distribuciones del dataset actual
if st.checkbox("Mostrar distribuciones del dataset"):
    try:
        data = pd.DataFrame(list(db["bank_clients"].find({}, {"_id": 0})))
        if not data.empty:
            if "y" in data.columns:
                st.write("📊 Distribución de la variable objetivo")
                fig, ax = plt.subplots()
                sns.countplot(data=data, x="y", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ El dataset no tiene una columna 'y' para graficar la variable objetivo.")
            if "age" in data.columns:
                st.write("📊 Distribución de edades")
                fig, ax = plt.subplots()
                sns.histplot(data=data, x="age", bins=20, kde=True, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ El dataset no contiene la columna 'age'.")
        else:
            st.warning("⚠️ La colección 'bank_clients' está vacía.")
    except Exception as e:
        st.error(f"❌ Error al generar visualizaciones: {e}")