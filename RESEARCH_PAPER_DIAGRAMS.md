# Research Paper Diagrams for WBC Diagnostic System v2.0

## 🏗️ 1. System Architecture Overview

### Mermaid Diagram Code:
```mermaid
graph TB
    subgraph "Client Layer"
        UI[React Frontend<br/>Dashboard, Patient Management<br/>Fairness & Insights Pages]
        Browser[Browser Client]
    end
    
    subgraph "API Gateway Layer"
        Backend[Node.js + Express Server<br/>REST API<br/>JWT Authentication]
    end
    
    subgraph "ML Service Layer"
        ML[FastAPI Model Server v2.0<br/>Hybrid MAE + TabNet/LR<br/>Multi-Explainer System]
        Calibration[Uncertainty Calibration<br/>Temperature Scaling<br/>Isotonic Regression]
        Explain[Explainability Module<br/>SHAP/LIME/Integrated Gradients]
    end
    
    subgraph "Data Layer"
        MongoDB[(MongoDB Database<br/>Patients, Predictions<br/>Fairness Metrics)]
        Models[Model Storage<br/>TabMAE Weights<br/>Trained Classifiers]
    end
    
    Browser --> UI
    UI -->|HTTP/REST| Backend
    Backend -->|Authentication| MongoDB
    Backend -->|Patient Records| MongoDB
    Backend -->|Prediction Requests| ML
    Backend -->|Model Status| ML
    
    ML -->|Load Models| Models
    ML -->|Query Data| MongoDB
    ML -->|Explanation Request| Explain
    ML -->|Calibration| Calibration
    Explain -->|SHAP Values| ML
    Calibration -->|Calibrated Probabilities| ML
    ML -->|Results| Backend
    Backend -->|API Response| UI
    
    style UI fill:#e1f5ff
    style Backend fill:#fff4e1
    style ML fill:#f0e1ff
    style MongoDB fill:#e8f5e9
    style Models fill:#fce4ec
    style Calibration fill:#f3e5f5
    style Explain fill:#e3f2fd
```

**Research Paper Description:**
*"Figure 1: System Architecture - The proposed WBC diagnostic system employs a three-tier architecture. The client layer (React) provides an interactive dashboard with fairness auditing and model insights. The API gateway (Node.js) manages authentication and routes requests. The ML service (FastAPI) implements a hybrid Masked Autoencoder with TabNet architecture, augmented with multi-explainer explainability and uncertainty calibration. All data persists in MongoDB with separate storage for trained model artifacts."*

---

## 🔄 2. Hybrid ML Architecture (MAE + TabNet)

### Mermaid Diagram Code:
```mermaid
graph LR
    subgraph "Input Layer"
        Features[30 WBC Features<br/>Wisconsin Breast Cancer Dataset]
    end
    
    subgraph "Stage 1: Self-Supervised Learning"
        MAE[Masked Autoencoder<br/>Masking Ratio: 25%<br/>Encoder-Decoder Architecture]
        Embed[32-Dimensional<br/>Embeddings]
    end
    
    subgraph "Stage 2: Classification"
        TabNet{TabNet<br/>Available?}
        LR[Logistic Regression<br/>Standard Classifier]
        Classifier[Binary Classifier<br/>Benign/Malignant]
    end
    
    subgraph "Stage 3: Calibration & Uncertainty"
        MC[Monte Carlo Dropout<br/>Uncertainty Estimation]
        Cal[Temperature Scaling<br/>Isotonic Regression]
        Output[Calibrated Probability<br/>Confidence Interval<br/>Prediction Class]
    end
    
    Features -->|Feature Scaling| MAE
    MAE -->|Masked Reconstruction| Embed
    Embed -->|Embedded Representation| TabNet
    Embed -->|Embedded Representation| LR
    
    TabNet -->|Yes| Classifier
    LR -->|No| Classifier
    
    Classifier --> MC
    MC --> Cal
    Cal --> Output
    
    style Features fill:#e1f5ff
    style MAE fill:#fff4e1
    style Embed fill:#f0e1ff
    style TabNet fill:#e8f5e9
    style LR fill:#e8f5e9
    style Classifier fill:#fce4ec
    style MC fill:#f3e5f5
    style Cal fill:#f3e5f5
    style Output fill:#e3f2fd
```

**Research Paper Description:**
*"Figure 2: Hybrid ML Architecture - The proposed hybrid approach consists of three stages. Stage 1 employs a Masked Autoencoder (MAE) for self-supervised pre-training, learning robust 32-dimensional embeddings through masked feature reconstruction. Stage 2 performs classification using either TabNet (with attention mechanisms) or Logistic Regression on embeddings. Stage 3 applies Monte Carlo dropout for uncertainty quantification and temperature scaling/isotonic regression for probability calibration, yielding trustworthy predictions with calibrated confidence intervals."*

---

## 🧠 3. Multi-Explainer Explainability Framework

### Mermaid Diagram Code:
```mermaid
graph TD
    subgraph "Input"
        Pred[Model Prediction<br/>+ Embedding]
        Features[30 WBC Features]
    end
    
    subgraph "Explainability Methods"
        SHAP[SHAP Explainer<br/>KernelExplainer<br/>Feature Attribution]
        LIME[LIME Explainer<br/>Local Interpretability<br/>Tabular Data]
        IG[Integrated Gradients<br/>Gradient-Based<br/>Gradient Accumulation]
    end
    
    subgraph "Unified Interface"
        UI[Unified Explainer<br/>Common API]
    end
    
    subgraph "Output"
        Imp1[Feature Importance<br/>30-Dimensional Vector]
        Imp2[Top-K Features<br/>Ranked List]
        Vis[Visualization<br/>Bar Charts & Force Plots]
    end
    
    Pred --> SHAP
    Pred --> LIME
    Pred --> IG
    
    Features --> SHAP
    Features --> LIME
    Features --> IG
    
    SHAP --> UI
    LIME --> UI
    IG --> UI
    
    UI --> Imp1
    UI --> Imp2
    Imp1 --> Vis
    Imp2 --> Vis
    
    style Pred fill:#e1f5ff
    style Features fill:#e1f5ff
    style SHAP fill:#fff4e1
    style LIME fill:#f0e1ff
    style IG fill:#e8f5e9
    style UI fill:#fce4ec
    style Imp1 fill:#f3e5f5
    style Imp2 fill:#f3e5f5
    style Vis fill:#e3f2fd
```

**Research Paper Description:**
*"Figure 3: Multi-Explainer Framework - We implement three complementary explainability methods: (1) SHAP provides Shapley values for feature attribution based on cooperative game theory, (2) LIME offers local linear approximations for individual predictions, and (3) Integrated Gradients compute gradient-based attributions through path integration. All methods operate through a unified interface, mapping embedding-level explanations back to original feature space and providing ranked feature importance for clinical interpretability."*

---

## 📊 4. Fairness Auditing Pipeline

### Mermaid Diagram Code:
```mermaid
graph TB
    subgraph "Data Collection"
        DB[(MongoDB<br/>Patient Records<br/>Demographics)]
        Meta[Metadata<br/>Age, Gender<br/>Clinical Features]
    end
    
    subgraph "Prediction Pipeline"
        Model[Model Prediction<br/>Binary Classification]
        Results[Prediction Results<br/>Benign/Malignant]
    end
    
    subgraph "Group Stratification"
        AgeGroup[Age Groups<br/>&lt;40, 40-60, &gt;60]
        GenderGroup[Gender Groups<br/>M, F, Other]
    end
    
    subgraph "Fairness Metrics"
        Acc[Accuracy<br/>Group-wise]
        Prec[Precision<br/>Group-wise]
        Recall[Recall<br/>Group-wise]
        F1[F1-Score<br/>Group-wise]
        DP[Demographic Parity<br/>Difference]
    end
    
    subgraph "Visualization"
        Chart1[Bar Charts<br/>Performance Comparison]
        Chart2[Fairness Dashboard<br/>Real-time Monitoring]
        Alert[Bias Alerts<br/>Threshold &gt; 0.1]
    end
    
    DB --> Meta
    Meta --> Model
    Model --> Results
    Results --> AgeGroup
    Results --> GenderGroup
    
    AgeGroup --> Acc
    AgeGroup --> Prec
    AgeGroup --> Recall
    AgeGroup --> F1
    
    GenderGroup --> Acc
    GenderGroup --> Prec
    GenderGroup --> Recall
    GenderGroup --> F1
    
    Acc --> DP
    Prec --> DP
    Recall --> DP
    F1 --> DP
    
    DP --> Chart1
    DP --> Chart2
    DP --> Alert
    
    style DB fill:#e8f5e9
    style Meta fill:#e1f5ff
    style Model fill:#fff4e1
    style Results fill:#f0e1ff
    style AgeGroup fill:#fce4ec
    style GenderGroup fill:#fce4ec
    style Acc fill:#f3e5f5
    style Prec fill:#f3e5f5
    style Recall fill:#f3e5f5
    style F1 fill:#f3e5f5
    style DP fill:#e3f2fd
    style Chart1 fill:#e1f5ff
    style Chart2 fill:#e1f5ff
    style Alert fill:#ffebee
```

**Research Paper Description:**
*"Figure 4: Fairness Auditing Pipeline - Our system implements comprehensive fairness monitoring by stratifying predictions across demographic groups (age: <40, 40-60, >60 years; gender: M, F, Other). Group-wise performance metrics (accuracy, precision, recall, F1) are computed alongside demographic parity differences. Real-time visualization through interactive dashboards enables continuous monitoring, with automated alerts triggered when demographic parity exceeds acceptable thresholds (>0.1), ensuring ethical AI deployment in clinical settings."*

---

## 🔄 5. End-to-End Prediction Flow

### Mermaid Diagram Code:
```mermaid
sequenceDiagram
    participant C as Clinician
    participant UI as React Frontend
    participant BE as Node.js Backend
    participant ML as ML Service
    participant DB as MongoDB
    participant F as Fairness Engine
    
    C->>UI: Create Patient + WBC Features
    UI->>BE: POST /api/patients
    BE->>DB: Store Patient Record
    DB-->>BE: Patient ID
    BE-->>UI: Patient Created
    
    C->>UI: Run Prediction + Select Explainer
    UI->>BE: POST /api/predictions/predict<br/>{patientId, method}
    BE->>DB: Fetch Patient Features
    DB-->>BE: Patient Data
    BE->>ML: POST /predict<br/>{features, method, age, gender}
    
    ML->>ML: Generate MAE Embeddings
    ML->>ML: TabNet/LR Classification
    ML->>ML: Monte Carlo Uncertainty
    ML->>ML: Temperature Calibration
    ML->>ML: SHAP/LIME/IG Explanation
    
    ML-->>BE: Prediction Response<br/>{result, calibrated_prob, uncertainty, shap_values}
    
    BE->>DB: Store Prediction Record
    DB-->>BE: Saved
    BE->>F: Compute Fairness Metrics
    F-->>BE: Group-wise Stats
    BE-->>UI: Complete Prediction
    
    UI->>UI: Display Result<br/>+ Visualization
    UI->>C: Show Calibrated Probability<br/>+ Feature Importance<br/>+ Fairness Status
    
    Note over C,DB: Continuous Learning Pipeline
    C->>UI: Verify Prediction (Optional)
    UI->>DB: Mark as verified
```

**Research Paper Description:**
*"Figure 5: End-to-End Prediction Workflow - The clinical prediction workflow: (1) clinician inputs patient WBC features, (2) system generates hybrid embeddings and classifications with uncertainty quantification, (3) multi-explainer module produces feature attributions, (4) fairness engine computes demographic parity, (5) calibrated probabilities and explanations are returned to clinician. The system supports continuous learning through verified prediction flags, enabling future model retraining on human-verified cases."*

---

## 🎓 6. Training Pipeline

### Mermaid Diagram Code:
```mermaid
graph TD
    subgraph "Dataset"
        Data[WBCD<br/>569 Samples<br/>30 Features<br/>2 Classes]
        Split[Train/Val/Test<br/>70/10/20 Split]
    end
    
    subgraph "Stage 1: MAE Pre-training"
        Pretrain[80 Epochs<br/>Learning Rate: 1e-3<br/>Batch Size: 32]
        MAEModel[TabMAE<br/>Encoder + Decoder<br/>Embed Dim: 32]
        Embeddings[Learned Embeddings<br/>Self-Supervised]
    end
    
    subgraph "Stage 2: Classifier Training"
        CLFTrain[TabNet/LR Training<br/>On Embeddings]
        CLFModel[Trained Classifier<br/>Weights Saved]
    end
    
    subgraph "Stage 3: Calibration"
        IsoTrain[Isotonic Regression<br/>5-Fold CV<br/>Temperature Scaling]
        CalModel[Calibrated Classifier<br/>Reliability Plot]
    end
    
    subgraph "Stage 4: Validation"
        Eval[Metrics<br/>Accuracy, AUC,<br/>Precision, Recall]
        Fair[Fairness Tests<br/>Demographic Parity]
        Save[Model Artifacts<br/>Saved to Disk]
    end
    
    Data --> Split
    Split --> Pretrain
    Pretrain --> MAEModel
    MAEModel --> Embeddings
    
    Embeddings --> CLFTrain
    CLFTrain --> CLFModel
    
    CLFModel --> IsoTrain
    IsoTrain --> CalModel
    
    CalModel --> Eval
    CalModel --> Fair
    Eval --> Save
    Fair --> Save
    
    style Data fill:#e1f5ff
    style Split fill:#fff4e1
    style Pretrain fill:#f0e1ff
    style MAEModel fill:#e8f5e9
    style Embeddings fill:#fce4ec
    style CLFTrain fill:#f3e5f5
    style CLFModel fill:#e3f2fd
    style IsoTrain fill:#ffebee
    style CalModel fill:#f1f8e9
    style Eval fill:#e0f2f1
    style Fair fill:#e0f2f1
    style Save fill:#fef5e7
```

**Research Paper Description:**
*"Figure 6: Model Training Pipeline - Our training procedure employs a four-stage pipeline: (1) Self-supervised MAE pre-training on all features for 80 epochs with 25% masking ratio, (2) Supervised classifier training using TabNet or Logistic Regression on learned 32-dimensional embeddings, (3) Probability calibration via 5-fold cross-validated isotonic regression, and (4) Comprehensive validation including performance metrics and fairness audits across demographic groups. All model artifacts are persisted for reproducible deployment."*

---

## 📈 7. System Components Interaction

### Mermaid Diagram Code:
```mermaid
graph LR
    subgraph "Frontend Services"
        D[Dashboard<br/>Statistics]
        P[Patients<br/>Management]
        F[Fairness<br/>Auditor]
        MI[Model Insights<br/>t-SNE Visualizations]
    end
    
    subgraph "Backend Services"
        Auth[JWT Auth<br/>Middleware]
        PatientsAPI[Patient CRUD<br/>API Endpoints]
        PredAPI[Prediction<br/>API]
        FairAPI[Fairness<br/>Metrics API]
    end
    
    subgraph "ML Services"
        Predict[Prediction<br/>Service]
        Explain[Explainer<br/>Service]
        Calibrate[Calibration<br/>Service]
        Embed[Embedding<br/>Service]
    end
    
    subgraph "Data Services"
        PatientDB[(Patient<br/>Database)]
        PredDB[(Prediction<br/>Database)]
        ModelStore[(Model<br/>Storage)]
    end
    
    D --> Auth
    P --> Auth
    F --> Auth
    MI --> Auth
    
    Auth --> PatientsAPI
    Auth --> PredAPI
    Auth --> FairAPI
    
    PatientsAPI --> PatientDB
    PredAPI --> Predict
    FairAPI --> Explain
    
    Predict --> Calibrate
    Predict --> Explain
    Predict --> Embed
    Predict --> ModelStore
    
    PredAPI --> PredDB
    Embed --> ModelStore
    
    style D fill:#e1f5ff
    style P fill:#e1f5ff
    style F fill:#e1f5ff
    style MI fill:#e1f5ff
    style Auth fill:#fff4e1
    style PatientsAPI fill:#fff4e1
    style PredAPI fill:#fff4e1
    style FairAPI fill:#fff4e1
    style Predict fill:#f0e1ff
    style Explain fill:#f0e1ff
    style Calibrate fill:#f0e1ff
    style Embed fill:#f0e1ff
    style PatientDB fill:#e8f5e9
    style PredDB fill:#e8f5e9
    style ModelStore fill:#e8f5e9
```

**Research Paper Description:**
*"Figure 7: System Component Architecture - The distributed architecture separates concerns across four layers: Frontend (React components for dashboard, patient management, fairness monitoring, and model insights), Backend (RESTful API with JWT authentication), ML Services (specialized modules for prediction, explanation, calibration, and embedding generation), and Data Services (MongoDB collections for patients, predictions, and model artifacts). This modular design enables independent scaling and maintenance of each component."*

---

## 🚀 8. Deployment Architecture

### Mermaid Diagram Code:
```mermaid
graph TB
    subgraph "Docker Infrastructure"
        subgraph "Container 1"
            ML[ML Service Container<br/>FastAPI + Python<br/>Port 8000]
        end
        
        subgraph "Container 2"
            BE[Backend Container<br/>Node.js + Express<br/>Port 4000]
        end
        
        subgraph "Container 3"
            FE[Frontend Container<br/>React Dev Server<br/>Port 3000]
        end
        
        subgraph "Container 4"
            DB[(MongoDB Container<br/>Database<br/>Port 27017)]
        end
        
        subgraph "Volumes"
            Models[Model Storage<br/>Persistent Volume]
            Data[MongoDB Data<br/>Persistent Volume]
        end
    end
    
    subgraph "External Access"
        User[Clinician<br/>Browser]
        API[API Endpoints<br/>Public/Private]
    end
    
    User -->|HTTP| FE
    User -->|REST| BE
    FE -->|Internal| BE
    BE -->|Internal| ML
    BE -->|Internal| DB
    
    ML --> Models
    DB --> Data
    
    style ML fill:#f0e1ff
    style BE fill:#fff4e1
    style FE fill:#e1f5ff
    style DB fill:#e8f5e9
    style Models fill:#fce4ec
    style Data fill:#e0f2f1
    style User fill:#ffebee
```

**Research Paper Description:**
*"Figure 8: Containerized Deployment - The system is deployed using Docker Compose orchestration with four independent containers: ML service (FastAPI), backend (Node.js), frontend (React), and database (MongoDB). Persistent volumes ensure model artifacts and database records survive container restarts. Internal networking enables secure service-to-service communication while exposing only necessary ports (3000, 4000, 8000) to external access. This containerized approach ensures reproducibility and simplifies scaling."*

---

## 📝 Quick Reference for Draw.io

### Color Scheme
- **Frontend/UI**: `#e1f5ff`
- **Backend/API**: `#fff4e1`
- **ML Services**: `#f0e1ff`
- **Database**: `#e8f5e9`
- **Model Storage**: `#fce4ec`
- **Special Services**: `#f3e5f5`
- **Visualization**: `#e3f2fd`
- **Alerts**: `#ffebee`

### Shape Recommendations
- **Rectangles**: Services, Components
- **Cylinders**: Databases, Storage
- **Diamonds**: Decision points
- **Parallelograms**: Input/Output
- **Rounded rectangles**: Processes

---

## 📊 Suggested Figure Numbers for Paper

1. **Figure 1**: System Architecture Overview
2. **Figure 2**: Hybrid ML Architecture (MAE + TabNet)
3. **Figure 3**: Multi-Explainer Framework
4. **Figure 4**: Fairness Auditing Pipeline
5. **Figure 5**: End-to-End Prediction Sequence
6. **Figure 6**: Training Pipeline
7. **Figure 7**: Component Interaction Architecture
8. **Figure 8**: Deployment Infrastructure

Each figure can be copied into Draw.io using the Mermaid code or recreated manually with the structure provided.

