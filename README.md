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

### Security Notes

- All medical data should be handled according to HIPAA/GDPR regulations
- Local processing ensures data privacy
- Vector database stores only embeddings, not raw patient data

