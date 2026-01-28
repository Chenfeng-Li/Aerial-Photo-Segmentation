FROM python:3.11-slim

WORKDIR /app

# If you later need system deps (opencv, etc.), add them here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code + checkpoints into the container
COPY . .

# Cloud Run listens on 8080
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
