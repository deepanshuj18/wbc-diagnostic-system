#!/bin/bash
# ml/train_and_save.sh
# Train the MAE model and save artifacts
echo "Training Masked Autoencoder..."
python train_mae.py --epochs 80 --embed_dim 32 --mask_ratio 0.25
echo "Training complete. Models saved to ./models/"






