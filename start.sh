#!/bin/bash
# Script de arranque para correr API + Dashboard en Elastic Beanstalk

# Activa virtualenv si es necesario
# source /var/app/venv/*/bin/activate  # EB lo hace automáticamente normalmente

# Inicia FastAPI en segundo plano
nohup uvicorn main:app --host 127.0.0.1 --port 8000 > api.log 2>&1 &

# Espera un poco para asegurar que el backend arranque
sleep 5

# Inicia Streamlit (usa la ruta relativa correcta)
streamlit run dashboard/app_streamlit.py --server.port 8080 --server.address 0.0.0.0
