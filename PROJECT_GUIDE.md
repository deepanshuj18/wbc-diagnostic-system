# WBC Diagnostic System v2.0 - Complete Project Guide

A research-grade full-stack web application for breast cancer prediction using hybrid MAE+TabNet architecture with multi-explainer explainability, fairness auditing, and uncertainty calibration.

---

## 🎯 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git (optional, for cloning)

### Start the Project

**Option 1: Using start script (Windows)**
```bash
start.bat
```

**Option 2: Using Docker Compose directly**
```bash
cd infra
docker-compose up --build
```

**Option 3: Background mode**
```bash
cd infra
docker-compose up -d --build
```

### Access the Application

After services start (5-10 minutes for first-time ML training):

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:4000
- **ML Service**: http://localhost:8000

### First Steps

1. Register an account at http://localhost:3000
2. Log in with your credentials
3. Create a patient in "Patients" tab
4. Add WBC features and run prediction
5. View results in Dashboard

---

## 🏗️ System Architecture

### Components

**Frontend (React)**
- Dashboard with statistics
- Patient management
- Fairness auditing dashboard
- Model insights with t-SNE visualizations

**Backend (Node.js + Express)**
- RESTful API
- JWT authentication
- Patient and prediction management
- Fairness metrics endpoint

**ML Service (FastAPI + Python)**
- Hybrid MAE + TabNet architecture
- Multi-explainer system (SHAP, LIME, IG)
- Uncertainty calibration
- Fairness computation

**Database (MongoDB)**
- Patient records
- Predictions history
- Model metadata

### Tech Stack

- **Frontend**: React 18, Axios, Plain CSS
- **Backend**: Node.js, Express, Mongoose, JWT
- **ML**: PyTorch, scikit-learn, FastAPI, SHAP, LIME
- **Database**: MongoDB 6.0
- **Deployment**: Docker Compose

---

## 📁 Project Structure

```
Major project/
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── pages/        # Dashboard, Patients, Fairness, Insights
│   │   ├── components/   # Navbar, SHAPPlot
│   │   ├── api.js        # API client
│   │   └── App.js        # Main app
│   └── package.json
├── backend/               # Node.js backend
│   ├── controllers/      # Business logic
│   ├── models/           # Mongoose schemas
│   ├── routes/           # API routes
│   ├── middleware/       # Auth middleware
│   └── package.json
├── ml/                    # Machine learning service
│   ├── train_v2.py       # Enhanced training with TabNet
│   ├── model_server_v2.py # FastAPI server
│   ├── explainers.py     # Multi-explainer module
│   ├── train_mae.py      # Basic MAE training
│   ├── utils.py          # Data utilities
│   └── requirements.txt
├── infra/                 # Infrastructure
│   └── docker-compose.yml # Container orchestration
├── start.bat             # Quick start script (Windows)
└── PROJECT_GUIDE.md     # This file
```

---

## 🚀 Advanced Usage

### Custom Training

Train with TabNet:
```bash
cd ml
python train_v2.py --epochs 80 --embed_dim 32 --use-tabnet
```

Train without TabNet (faster):
```bash
cd ml
python train_v2.py --epochs 80 --embed_dim 32
```

### Manual Service Start

**Start MongoDB:**
```bash
docker run -d -p 27017:27017 --name mongo mongo:6.0
```

**Start ML Service:**
```bash
cd ml
uvicorn model_server_v2:app --host 0.0.0.0 --port 8000
```

**Start Backend:**
```bash
cd backend
npm install
npm start
```

**Start Frontend:**
```bash
cd frontend
npm install
npm start
```

### API Endpoints

**Authentication**
- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login

**Patients**
- `GET /api/patients` - List patients
- `POST /api/patients` - Create patient
- `PUT /api/patients/:id` - Update patient
- `DELETE /api/patients/:id` - Delete patient

**Predictions**
- `POST /api/predictions/predict` - Run prediction
- `GET /api/predictions` - List predictions

**Models**
- `GET /api/models/status` - Model status
- `GET /api/models/fairness` - Fairness metrics
- `GET /api/models/embeddings` - t-SNE visualization

**ML Service (v2)**
- `POST /predict` - Enhanced prediction
- `POST /explain` - Multi-explainer explanations
- `GET /model_fairness` - Fairness metrics
- `POST /embeddings_visualization` - t-SNE data

---

## 🔧 Configuration

### Environment Variables

**Note:** `.env.example` file in root serves as template for Docker deployment.

**Root .env.example (For Docker)**
```bash
# Mongo
MONGO_URI=mongodb://mongo:27017/wbcdb

# Backend
JWT_SECRET=supersecret_jwt

# Model API (internal docker network)
MODEL_API_URL=http://ml:8000

# Other
PORT=4000
```

**For Local Development:**

Create `backend/.env`:
```bash
MONGO_URI=mongodb://localhost:27017/wbc-db
JWT_SECRET=your-secret-key-here
PORT=4000
MODEL_API_URL=http://localhost:8000
```

Create `frontend/.env`:
```bash
REACT_APP_API_URL=http://localhost:4000/api
```

**ML Service Environment Variable:**
```bash
MODEL_DIR=models
```

---

## 📊 Features

### Core Capabilities

1. **Hybrid ML Architecture**
   - Masked Autoencoder for self-supervised learning
   - TabNet or Logistic Regression for classification
   - 32-dimensional embeddings

2. **Multi-Explainer System**
   - SHAP: Shapley additive explanations
   - LIME: Local interpretable model-agnostic
   - Integrated Gradients: Gradient-based attribution

3. **Uncertainty Calibration**
   - Monte Carlo dropout
   - Temperature scaling
   - Isotonic regression
   - Calibrated confidence intervals

4. **Fairness Auditing**
   - Demographic parity monitoring
   - Age and gender group analysis
   - Automated bias detection
   - Real-time dashboard

5. **Visualizations**
   - t-SNE embedding plots
   - SHAP force plots
   - Fairness bar charts
   - Model architecture display

---

## 🧪 Testing

### Test ML Service
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

### Check Model Status
```bash
curl http://localhost:8000/model_status
```

### Verify Services
```bash
docker ps  # Should show 4 containers running
```

---

## 🐛 Troubleshooting

### Services Not Starting

**Problem**: Docker containers fail to start
**Solution**: 
```bash
cd infra
docker-compose down -v  # Clean restart
docker-compose up --build
```

### Frontend Not Loading

**Problem**: White screen or connection errors
**Solution**:
- Wait 1-2 minutes for React dev server to compile
- Check `docker logs frontend`
- Hard refresh browser: Ctrl + Shift + F5

### ML Model Not Ready

**Problem**: "Models not loaded" error
**Solution**:
- Check training completed: `docker logs ml`
- Verify models directory exists: `docker exec ml ls -la models/`
- Retrain if needed: `docker-compose restart ml`

### Database Connection Issues

**Problem**: Backend can't connect to MongoDB
**Solution**:
- Verify MongoDB running: `docker ps | grep mongo`
- Check MONGO_URI in docker-compose.yml
- Restart backend: `docker-compose restart backend`

---

## 📈 Performance

**Model Performance (WBCD Dataset)**
- Accuracy: ~96-98%
- AUC-ROC: ~0.99
- Precision: ~97-99%
- Recall: ~95-97%

**System Requirements**
- RAM: 8GB+ recommended
- Disk: 2GB+ for Docker images
- CPU: Multi-core recommended for training
- GPU: Optional (CPU works fine)

**Training Time**
- Without TabNet: ~5-10 minutes
- With TabNet: ~15-20 minutes
- First build: ~30 minutes (downloading images)

---

## 🔒 Security & Privacy

- JWT authentication for API access
- Password hashing with bcrypt
- CORS enabled for frontend
- Environment variables for secrets
- Database access control

**Production Deployment**
- Change JWT_SECRET
- Use MongoDB Atlas or secured instance
- Enable HTTPS
- Configure firewall rules
- Regular backups

---

## 🌐 Production Deployment

### Using Docker Compose

1. Update environment variables
2. Build images: `docker-compose build`
3. Start services: `docker-compose up -d`
4. Set up reverse proxy (nginx/traefik)
5. Configure SSL certificates

### Cloud Platforms

**Render.com**: Deploy each service separately
**AWS ECS**: Use Docker containers
**Heroku**: Use container builds
**Azure Container Instances**: Simple deployment

---

## 📚 Research & Academic Use

### Paper Citations

If using this system in research, cite:
- Wisconsin Breast Cancer Dataset
- SHAP library
- TabNet model
- PyTorch framework

### Dataset

The system uses Wisconsin Breast Cancer (Diagnostic) Dataset:
- 569 samples
- 30 features
- 2 classes (Benign, Malignant)
- Public domain

---

## 🆘 Support & Issues

### Getting Help

1. Check service logs: `docker logs <service-name>`
2. Verify environment variables
3. Check Docker resources
4. Review this guide

### Common Commands

```bash
# View all logs
docker-compose logs -f

# Restart specific service
docker-compose restart frontend

# Stop all services
docker-compose down

# Clean restart (removes volumes)
docker-compose down -v && docker-compose up --build

# Check container status
docker ps

# Execute commands in container
docker exec -it frontend sh
```

---

## 📝 License & Credits

**Built with:**
- React, Node.js, FastAPI, PyTorch, MongoDB
- Wisconsin Breast Cancer Dataset
- SHAP, LIME, TabNet libraries

**For academic use and research purposes.**

---

## 🎯 Summary

**Quick Start Checklist:**
- [ ] Docker installed
- [ ] Clone/download project
- [ ] Run `start.bat` or `docker-compose up`
- [ ] Wait 5-10 minutes
- [ ] Open http://localhost:3000
- [ ] Register and start using

**That's it! Your WBC Diagnostic System v2.0 is ready to use.**

---

## 🧠 How the ML System Works (Detailed)

### Architecture Overview

**Three-Stage Pipeline:**
```
WBC Features (30) 
    → Stage 1: Masked Autoencoder (MAE)
    → Embeddings (32-dim)
    → Stage 2: TabNet or Logistic Regression
    → Raw Probability
    → Stage 3: Temperature Scaling
    → Calibrated Probability + Uncertainty
```

### Step-by-Step ML Workflow

**Training Phase (`ml/train_v2.py`):**

1. **Phase 1: Self-Supervised Learning (MAE)**
   - Takes 30 WBC features
   - Randomly masks 25% of features
   - Encoder learns robust representations
   - Decoder reconstructs masked features
   - Trains for 80 epochs

2. **Phase 2: Classification**
   - 32D embeddings fed to classifier
   - TabNet (attention) or Logistic Regression
   - Learns to predict Benign/Malignant
   - Evaluates on validation set

3. **Phase 3: Calibration**
   - Calibrates raw model outputs
   - Ensures probabilities reflect true confidence
   - Uses 5-fold cross-validation

**Inference Phase (`ml/model_server_v2.py`):**

When you send prediction request:

```
1. Receive: 30 WBC features + age + gender + explainer method

2. Load Models:
   - TabMAE encoder/decoder
   - Classifier (TabNet or LR)
   - Calibrated classifier
   - Scaler for normalization

3. Generate Embeddings:
   features → normalize → MAE encoder → 32D embedding

4. Classify:
   embedding → classifier → raw probability

5. Calibrate:
   raw probability → calibrated classifier → calibrated probability

6. Uncertainty:
   Run 10 MC Dropout iterations → compute std → uncertainty measure

7. Explain:
   Select method (SHAP/LIME/IG) → generate feature importance

8. Return:
   - Prediction (0=Benign, 1=Malignant)
   - Calibrated probability
   - Uncertainty score
   - SHAP values
   - Embedding representation
```

### Backend → ML Integration

**Flow:**
```
Patient Features → Backend → FastAPI ML Service → Results → Database → Frontend
```

**Backend Code:**
```javascript
const response = await axios.post('http://ml:8000/predict', {
  features: [30 float values],  // WBC features
  method: 'shap',               // Explainer method
  age: patient.age,            // For fairness
  gender: patient.gender       // For fairness
});
```

**ML Service Code:**
```python
@app.post("/predict")
def predict(req: PredictRequest):
    x = scaler.transform([req.features])
    _, z, _ = model(torch.from_numpy(x))
    prob = calibrated_clf.predict_proba(z)[:, 1]
    shap_vals = explainer.explain(x, method=req.method)
    return {
        "result": "Malignant" if prob > 0.5 else "Benign",
        "calibrated_probability": prob,
        "shap_values": shap_vals
    }
```

### Model Inputs/Outputs

**Input:**
- `features`: Array of 30 floats (WBC features)
- `method`: 'shap' | 'lime' | 'ig' (optional)
- `age`: Integer (optional)
- `gender`: String (optional)

**Output:**
```json
{
  "prediction": 0,
  "result": "Benign",
  "calibrated_probability": 0.78,
  "uncertainty": 0.08,
  "explanation_method": "SHAP",
  "shap_values": [0.1, -0.05, ...],
  "embedding_mean": [0.2, -0.1, ...],
  "model_version": "2.0.0"
}
```

### Key ML Concepts

- **MAE**: Masked Autoencoder learns feature relationships by reconstructing masked inputs
- **TabNet**: Attention-based tabular model for better interpretability
- **Calibration**: Ensures 80% confidence means 80% accuracy
- **MC Dropout**: Multiple runs with randomness to estimate uncertainty
- **SHAP/LIME/IG**: Different methods to explain which features matter

---

---

## 🎓 Presentation Guide - Explaining to Teachers

### Quick System Explanation

**What This System Does:**
A web-based AI that helps doctors diagnose breast cancer from 30 numerical patient measurements.

**How It Predicts:**
1. **Self-Supervised Learning (MAE)**: Learns feature patterns by reconstructing masked inputs
2. **Classification (TabNet/LR)**: Converts learned patterns into benign/malignant prediction  
3. **Calibration**: Adjusts probabilities to be trustworthy for clinical decisions

**Why It's Special:**
- **Explains Every Prediction**: Shows which features drove the decision (SHAP/LIME/IG)
- **Quantifies Uncertainty**: Tells you how confident it is
- **Monitors Fairness**: Ensures no demographic bias
- **Production-Ready**: Fully deployed with Docker

**Real-World Impact:**
- Doctor enters 30 measurements → Gets instant second opinion with reasoning
- Sees confidence level and feature importance
- Can verify the prediction makes sense
- Trustworthy AI for clinical decisions

---

**See `TEACHER_PRESENTATION_GUIDE.md` for full presentation-ready explanation with scripts and talking points.**

