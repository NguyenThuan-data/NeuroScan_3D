# NeuroScan-3D-RAG

## System Architecture Overview

NeuroScan-3D-RAG is an advanced medical imaging analysis system that combines 3D segmentation with Retrieval-Augmented Generation (RAG) for comprehensive neuroimaging analysis and protocol-guided diagnosis.

### Architecture Components

#### 1. **Data Layer** (`data/`)
- **`dicom_inputs/`**: Raw DICOM (.dcm) files from medical imaging devices
- **`hospital_knowledge/`**: PDF protocols, case studies, and institutional medical knowledge (The "Internal Truth")
- **`vector_db/`**: Local vector database storage (Pinecone/Milvus) for semantic search

#### 2. **Models Layer** (`models/`)
- **`nnunet_weights/`**: Pre-trained nnU-Net weights for 3D medical image segmentation
- **`medical_clip/`**: Fine-tuned Medical CLIP model for vision-text alignment in medical contexts

#### 3. **Source Code** (`src/`)
- **`app.py`**: Frontend application (Streamlit/React) for user interaction
- **`inference.py`**: 3D analysis logic implementing nnU-Net segmentation with Test-Time Augmentation (TTA)
- **`rag_engine.py`**: Retrieval engine implementing GraphRAG/Vector search for knowledge retrieval
- **`llm_core.py`**: Local LLM wrapper (Ollama/Llama3) for generating medical reports and insights

#### 4. **Configuration** (`config.yaml`)
- Thresholds for segmentation and analysis
- File paths and data directories
- Security settings and access controls

### Workflow

1. **Input**: DICOM files are loaded from `data/dicom_inputs/`
2. **Segmentation**: `inference.py` processes images using nnU-Net with TTA
3. **Knowledge Retrieval**: `rag_engine.py` queries hospital knowledge base for relevant protocols
4. **Analysis**: `llm_core.py` generates comprehensive reports combining segmentation results with retrieved knowledge
5. **Output**: Results displayed through `app.py` frontend

### Setup Instructions

1. Install dependencies (see requirements.txt)
2. Configure paths in `config.yaml`
3. Load model weights into `models/` directories
4. Initialize vector database in `data/vector_db/`
5. Run `streamlit run src/app.py` or start the React frontend

---

## 🧠 Tumor Vision API (Brain Tumor Classification)

**Tumor Vision API** is a FastAPI + Streamlit application for brain tumor classification and segmentation using a Multi-Task U-Net model.

### 🌐 Live Deployment

**Production Application:**
- **Frontend UI:** Streamlit Cloud
- **Backend API:** Hugging Face Spaces (16GB RAM, free tier)
- **Architecture:** Distributed system with HTTPS communication

**Access the App:**
- Visit the deployed Streamlit application
- Upload a ZIP file containing 4 NIfTI files (flair, t1, t1ce, t2)
- Get instant classification (HGG/LGG) and segmentation results

### 💻 Local Development

#### 1. Start the API Server
```bash
source venv_api/Scripts/activate
uvicorn tumor_vision_api.api_logic:app --reload --port 8000
```

#### 2. Start the Streamlit UI
```bash
streamlit run streamlit_app.py
```

#### 3. Use the Application
1. Open http://localhost:8501
2. Upload a ZIP file containing 4 NIfTI files (flair, t1, t1ce, t2)
3. Click "Predict Tumor"
4. View classification results (HGG/LGG) and segmentation visualization

### ✨ Features
- ✅ Multi-Task U-Net for simultaneous classification and segmentation
- ✅ Interactive Streamlit UI with drag-and-drop upload
- ✅ Real-time prediction with progress notifications
- ✅ Interactive probability charts (Plotly)
- ✅ 4-panel tumor segmentation visualization
- ✅ RESTful API with FastAPI
- ✅ Automatic model loading on startup
- ✅ Cloud deployment with Hugging Face Spaces (16GB RAM)
- ✅ CORS-enabled for distributed architecture

### 🚀 Deployment Architecture

```
┌─────────────────────┐         HTTPS          ┌──────────────────────┐
│  Streamlit Cloud    │ ───────────────────────> │  Hugging Face Spaces │
│  (Frontend UI)      │    POST /predict_tumor/ │  (FastAPI + Model)   │
│  - File Upload      │ <─────────────────────── │  - 16GB RAM          │
│  - Visualization    │         JSON Response    │  - Docker Container  │
└─────────────────────┘                          └──────────────────────┘
```

### 🔌 API Endpoints
- `POST /predict_tumor/` - Upload ZIP with 4 NIfTI files, get classification & segmentation
- `POST /evaluate_tumor/` - Evaluate predictions against ground truth
- `GET /health` - Health check (model status, device info)
- `GET /` - API information
- `GET /docs` - Interactive Swagger documentation

### Security Notes

- All medical data should be handled according to HIPAA/GDPR regulations
- Local processing ensures data privacy
- Vector database stores only embeddings, not raw patient data

