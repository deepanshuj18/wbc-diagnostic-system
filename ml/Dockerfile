FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN mkdir -p models
EXPOSE 8000
CMD ["sh", "-c", "python train_v2.py --epochs 80 --embed_dim 32 --mask_ratio 0.25 || true && uvicorn model_server_v2:app --host 0.0.0.0 --port 8000"]
