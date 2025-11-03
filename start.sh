#!/bin/bash
# Script de arranque para correr API + Dashboard

# Ejecuta FastAPI en segundo plano
nohup python api/app.py > fastapi.log 2>&1 &

# Ejecuta Streamlit en primer plano en el puerto correcto
streamlit run c:/Users/anton/OneDrive/Documentos/bankmarketing_app/dashboard/app_streamlit.py --server.port $PORT --server.address 0.0.0.0
