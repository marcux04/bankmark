#!/bin/bash
# Inicia la API (FastAPI) en segundo plano
echo "Starting FastAPI..."
nohup gunicorn -w 1 -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:8000 &

# Espera unos segundos para asegurar que la API esté arriba
sleep 5

# Inicia el dashboard Streamlit en el puerto asignado por EB ($PORT)
echo "Starting Streamlit Dashboard..."
streamlit run dashboard/app_streamlit.py --server.port $PORT --server.address 0.0.0.0
