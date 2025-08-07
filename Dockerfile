FROM python:3.11-slim

# OS deps (xgboost → libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Installe les deps d'abord (meilleur cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY main.py /app/

# Exécution
CMD ["python", "main.py"]
