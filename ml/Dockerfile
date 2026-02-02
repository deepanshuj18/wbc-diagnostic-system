FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models
EXPOSE 8000
CMD ["sh", "-c", "python train_mae.py --epochs 80 --embed_dim 32 --mask_ratio 0.25 || echo 'Model training skipped' && uvicorn model_server:app --host 0.0.0.0 --port 8000 --workers 1"]
