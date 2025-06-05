#!/bin/bash

# Vérifie que le fichier .env existe
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Active l'environnement virtuel
source venv/bin/activate

# Lance l'application avec Gunicorn pour la production
exec gunicorn main:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker --log-level info --access-logfile -
