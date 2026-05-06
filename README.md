# 🧬 WBC Diagnostic System v2.0

An enterprise-grade, agentic AI diagnostic system for classifying Wisconsin Breast Cancer (WBC) tumors. This system features a robust hybrid machine learning architecture (Masked Autoencoders + TabNet/Logistic Regression), rigorous probability calibration, multi-method explainability (SHAP, LIME, Integrated Gradients), and an active fairness monitoring pipeline.

## 🌟 Key Features

*   **Hybrid ML Architecture:** Uses a PyTorch-based Masked Autoencoder (MAE) to generate rich embeddings from 30 tabular features, followed by a calibrated classifier.
*   **Deep Explainability (XAI):** Clinicians can view real-time feature importance using SHAP, LIME, or Integrated Gradients to understand *why* a prediction was made.
*   **Probability Calibration:** Uses Platt scaling (Sigmoid calibration) on a dedicated holdout set so that a 90% confidence score mathematically reflects a 90% true probability.
*   **Fairness Auditing:** Built-in demographic parity monitoring to ensure the model performs equally well across different patient groups.
*   **Secure & Isolated:** Clinician accounts are isolated with JWT authentication, ensuring users can only access their own patient records.

---

## 🏗️ System Architecture

The system is fully containerized using Docker and consists of three main microservices:

1.  **Frontend (`/frontend`)**: React.js SPA providing the clinician dashboard, patient management, explainability charts, and embedding visualizations (t-SNE).
2.  **Backend (`/backend`)**: Node.js/Express REST API that handles JWT authentication, patient data persistence in MongoDB, and proxies ML requests.
3.  **ML Service (`/ml`)**: Python/FastAPI service hosting the PyTorch and Scikit-Learn models, explainers, and fairness evaluators.
4.  **Database**: MongoDB container persisting all user, patient, and prediction data.

---

## 🚀 Quickstart

### Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
*   Git

### Running Locally

1.  Clone the repository and navigate to the root directory.
2.  Start the entire system using the provided batch script (Windows) or via Docker Compose directly:

**Using the batch script:**
```powershell
.\start.bat
```

**Using Docker Compose directly:**
```bash
cd infra
docker-compose up -d --build
```

### Accessing the Application
*   **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
*   **Backend API:** [http://localhost:4000](http://localhost:4000)
*   **ML API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Project Structure

```text
├── backend/                # Node.js Express API
│   ├── controllers/        # Route handlers (auth, patients, predictions)
│   ├── models/             # Mongoose schemas
│   └── routes/             # API routing
├── frontend/               # React User Interface
│   ├── src/
│   │   ├── components/     # Reusable UI components (SHAPPlot, etc.)
│   │   └── pages/          # Views (Dashboard, PatientDetail, Fairness)
├── infra/                  # Infrastructure & DevOps
│   └── docker-compose.yml  # Multi-container orchestration
├── ml/                     # Python ML Service
│   ├── explainers.py       # SHAP, LIME, and IG implementations
│   ├── model_server_v2.py  # FastAPI inference server
│   ├── train_v2.py         # Model training pipeline
│   ├── utils.py            # Data loading and 4-way splitting
│   └── Dockerfile          # ML container definition (CPU-optimized)
└── start.bat               # Windows startup helper script
```

---

## 🛠️ ML Training Pipeline

The ML pipeline (`train_v2.py`) is designed to prevent data leakage and ensure reliable metrics:

1.  **Data Split:** The data is split 4 ways: `Train`, `Validation` (for early stopping), `Calibration` (for Platt scaling), and `Test` (untouched until final evaluation).
2.  **Phase 1 (MAE):** A Masked Autoencoder is trained to reconstruct the tabular data with 25% random masking.
3.  **Phase 2 (Embeddings):** The MAE's encoder is used to generate rich, dense embeddings for the dataset.
4.  **Phase 3 (Classifier):** A Logistic Regression (or TabNet) classifier is trained on the embeddings.
5.  **Phase 4 (Calibration):** The classifier's probabilities are calibrated using the dedicated holdout set to ensure reliable confidence scores.

---

## 🔒 Security

*   **Authentication:** JWT-based authentication. The JWT secret is injected via environment variables.
*   **Role-Based Access:** Registration defaults to the `clinician` role to prevent unauthorized admin access.
*   **Data Isolation:** All patient and prediction endpoints strictly verify `createdBy` ownership against the authenticated token.

---

## 📝 License

This project is intended for educational and research purposes. Do not use for actual medical diagnosis without regulatory approval and clinical validation.
