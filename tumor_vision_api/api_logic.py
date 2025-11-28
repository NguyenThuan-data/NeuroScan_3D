import matplotlib.pyplot as plt
from PIL import Image   
import io
import base64
import numpy as np
import sys
import os
import torch
import tempfile
import zipfile
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Add project root to Python path so imports work when running uvicorn
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tumor_vision_api.model_utils import MultiTaskUNet, HybridLoss
from tumor_vision_api.data_preprocessing_utils import BratsDataset

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app_state = {} # Global dictionary to store model references (can hold multiple models)

# ============================================================
# PREPROCESSING FUNCTION 
# ============================================================
def preprocess_single_patient(patient_folder_path, target_shape=(128, 128, 128), return_ground_truth=False):
    """
    Load and preprocess a single patient using BratsDataset's existing logic.
    
    Args:
        patient_folder_path: Full path to patient folder (e.g., /data/BraTS20_Training_001)
        target_shape: Target dimensions for center crop
        return_ground_truth: If True, also return ground truth mask and grade (for evaluation)
    
    Returns:
        If return_ground_truth=False: img_tensor [1, 4, D, H, W]
        If return_ground_truth=True: (img_tensor, mask_tensor, grade_tensor)
    """
    # Normalize path separators
    patient_folder_path = os.path.normpath(patient_folder_path)
    
    # Auto-detect patient ID from files in the folder
    files = os.listdir(patient_folder_path)
    
    # Find a NIfTI file and extract patient ID from it
    patient_id = None
    for f in files:
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            # Extract patient ID: BraTS20_Training_001_flair.nii -> BraTS20_Training_001
            if '_flair' in f or '_t1' in f or '_t2' in f:
                patient_id = f.split('_flair')[0].split('_t1')[0].split('_t2')[0].split('_t1ce')[0]
                break
    
    if patient_id is None:
        # Fallback: use folder name as patient ID
        patient_id = os.path.basename(patient_folder_path)
    
    print(f"Auto-detected patient ID: {patient_id}")
    print(f"Patient folder: {patient_folder_path}")
    
    # Get parent directory and folder name separately
    root_dir = os.path.dirname(patient_folder_path)
    folder_name = os.path.basename(patient_folder_path)
    
    # Create a temporary dataset - use folder_name as the patient_id
    # because BratsDataset will join root_dir + patient_id
    temp_dataset = BratsDataset(
        patient_ids=[folder_name],  # Use the actual folder name
        root_dir=root_dir,
        mapping_df=None,
        target_shape=target_shape
    )
    
    # Use the existing __getitem__ logic to load and preprocess
    img_tensor, mask_tensor, grade_tensor = temp_dataset[0]
    
    # Add batch dimension [4, D, H, W] -> [1, 4, D, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    if return_ground_truth:
        mask_tensor = mask_tensor.unsqueeze(0)
        return img_tensor, mask_tensor, grade_tensor
    else:
        return img_tensor


# ============================================================
# HELPER FUNCTION: Load model dynamically
# ============================================================
def load_model(model_path, device):
    """Load a model from a given path."""
    model = MultiTaskUNet(in_channels=4, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


# ============================================================
# LIFESPAN CONTEXT - Loads default model on startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API STARTING: Loading default model into memory...")
    
    # Load the default model
    default_model = load_model(DEFAULT_MODEL_PATH, DEVICE)

    # Store models in a dictionary (supports multiple models)
    app_state["models"] = {
        "default": default_model
    }
    app_state["device"] = DEVICE
    
    print(f"Default model loaded on: {DEVICE}")
    yield # API is ready to serve requests
    
    # API Shutdown logic
    app_state.clear()
    print("API SHUTDOWN: Models unloaded.")

app = FastAPI(lifespan=lifespan)

# ============================================================
# CORS CONFIGURATION - Allow Streamlit Cloud to make requests
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.streamlit.app",  # Streamlit Cloud apps
        "http://localhost:8501",    # Local Streamlit development
        "http://localhost:3000",    # Alternative local port
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================
# REQUEST MODELS
# ============================================================
class PredictionRequest(BaseModel):
    patient_folder_path: str
    model_name: str = "default"  # Which model to use


class LoadModelRequest(BaseModel):
    model_path: str
    model_name: str  # Custom name for this model (e.g., "v2", "experimental")


# ============================================================
# PREDICTION ENDPOINT (FILE UPLOAD VERSION)
# ============================================================
@app.post("/predict_tumor/")
async def predict_tumor_analysis(
    zip_file: UploadFile = File(..., description="ZIP file containing 4 NIfTI modalities"),
    model_name: str = Form(default="default")
):
    """
    Predict brain tumor classification and segmentation from uploaded ZIP file.
    
    Args:
        zip_file: ZIP file containing flair, t1, t1ce, t2 NIfTI files
        model_name: Model to use for prediction (default: "default")
    
    Returns:
        JSON with diagnosis, probabilities, and segmentation visualization
    """
    temp_dir = None
    try:
        print(f"Received ZIP file: {zip_file.filename}")
        
        # Check if requested model exists
        if model_name not in app_state["models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(app_state['models'].keys())}"
            )
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {temp_dir}")
        
        # Save uploaded ZIP file
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            content = await zip_file.read()
            f.write(content)
        print(f"Saved ZIP file ({len(content)} bytes)")
        
        # Extract ZIP contents
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted ZIP to: {extract_dir}")
        
        # Find the folder containing NIfTI files
        # (handle case where ZIP has nested folder structure)
        patient_path = extract_dir
        for root, dirs, files in os.walk(extract_dir):
            nifti_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
            if len(nifti_files) >= 4:
                patient_path = root
                print(f"Found NIfTI files in: {patient_path}")
                break
        
        # Validate that we have the required files
        all_files = os.listdir(patient_path)
        nifti_files = [f for f in all_files if f.endswith(('.nii', '.nii.gz'))]
        
        if len(nifti_files) < 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 4 NIfTI files, found {len(nifti_files)}: {nifti_files}"
            )
        
        print(f"Found {len(nifti_files)} NIfTI files: {nifti_files}")
        
        # Get the requested model
        model = app_state["models"][model_name]
        
        # ==========================================
        # CRITICAL FIX: Use BratsDataset preprocessing
        # ==========================================
        print(f"Loading patient data from: {patient_path}")
        img_tensor = preprocess_single_patient(patient_path)
        img_tensor = img_tensor.to(app_state["device"])
        
        # Run inference
        print(f"Running inference with model '{model_name}'...")
        with torch.no_grad():
            seg_logits, class_logits = model(img_tensor)
        
        # Get classification result
        grade_prob = torch.sigmoid(class_logits).item()
        diagnosis = "HGG" if grade_prob > 0.5 else "LGG"
        
        print(f"Prediction: {diagnosis} (HGG probability: {grade_prob:.4f})")
        
        # Get predicted segmentation mask
        pred_mask_np = torch.argmax(
            torch.softmax(seg_logits, dim=1), dim=1
        ).cpu().numpy()[0]
        
        # Extract slices from different modalities for better visualization
        slice_idx = 64  # Mid-slice
        
        # Get all modality slices: [FLAIR, T1, T1CE, T2]
        flair_slice = img_tensor.cpu().numpy()[0, 0, :, :, slice_idx]
        t1_slice = img_tensor.cpu().numpy()[0, 1, :, :, slice_idx]
        t1ce_slice = img_tensor.cpu().numpy()[0, 2, :, :, slice_idx]
        t2_slice = img_tensor.cpu().numpy()[0, 3, :, :, slice_idx]
        
        # Normalize intensity for better visualization (scale to 0-1 range)
        def normalize_for_display(img_slice):
            """Normalize image slice to [0, 1] for better visualization"""
            img_min, img_max = img_slice.min(), img_slice.max()
            if img_max > img_min:
                return (img_slice - img_min) / (img_max - img_min)
            return img_slice
        
        flair_display = normalize_for_display(flair_slice)
        t1_display = normalize_for_display(t1_slice)
        t1ce_display = normalize_for_display(t1ce_slice)
        t2_display = normalize_for_display(t2_slice)
        
        # ==========================================
        # VISUALIZATION - Separate images for each tumor class
        # ==========================================
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        pred_mask_slice = pred_mask_np[:, :, slice_idx]
        
        # 1. Original MRI (T1 - Fluid dark, fat bright, anatomy clear)
        axes[0].imshow(t1_display, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('MRI (T1 - Anatomy)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. NCR/NET (Class 1) - Necrotic Core
        # Use FLAIR: shows abnormality bright but necrosis is dark on T1CE (no contrast uptake)
        axes[1].imshow(flair_display, cmap='gray', vmin=0, vmax=1)
        ncr_mask = np.ma.masked_where(pred_mask_slice != 1, pred_mask_slice)
        axes[1].imshow(ncr_mask, cmap='Reds', vmin=0.5, vmax=1.5, alpha=0.8, interpolation='nearest')
        axes[1].set_title('NCR/NET (Necrotic Core)\nFLAIR: Bright abnormal, Dark on T1CE', 
                         fontsize=12, fontweight='bold', color='darkred')
        axes[1].axis('off')
        
        # 3. ED (Class 2) - Edema (Swelling)
        # Use T2: fluid/edema appears BRIGHT (white), shows swelling clearly
        axes[2].imshow(t2_display, cmap='gray', vmin=0, vmax=1)
        ed_mask = np.ma.masked_where(pred_mask_slice != 2, pred_mask_slice)
        axes[2].imshow(ed_mask, cmap='YlOrBr', vmin=1.5, vmax=2.5, alpha=0.8, interpolation='nearest')
        axes[2].set_title('ED (Edema - Swelling)\nT2: Fluid BRIGHT, Not on T1CE', 
                         fontsize=12, fontweight='bold', color='orange')
        axes[2].axis('off')
        
        # 4. ET (Class 3) - Enhancing Tumor (Active Core)
        # Use T1CE: active tumor lights up BRIGHT (took up contrast agent)
        axes[3].imshow(t1ce_display, cmap='gray', vmin=0, vmax=1)
        et_mask = np.ma.masked_where(pred_mask_slice != 3, pred_mask_slice)
        axes[3].imshow(et_mask, cmap='Blues', vmin=2.5, vmax=3.5, alpha=0.8, interpolation='nearest')
        axes[3].set_title('ET (Enhancing Tumor - Active Core)\nT1CE: Tumor BRIGHT (contrast uptake)', 
                         fontsize=12, fontweight='bold', color='darkblue')
        axes[3].axis('off')
        
        # Add overall title
        fig.suptitle(f'Prediction: {diagnosis} (HGG Prob: {grade_prob:.2%})', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig) # Close to free memory

        # Encode image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")

        return {
            "status": "success",
            "model_used": model_name,
            "diagnosis": diagnosis,
            "hgg_probability": round(grade_prob, 4),
            "lgg_probability": round(1 - grade_prob, 4),
            "predicted_mask_image_b64": img_base64,
            "message": "Analysis complete with segmentation visualization."
        }

    except FileNotFoundError as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {
            "status": "error",
            "message": f"File not found: {str(e)}"
        }
    except Exception as e:
        import traceback
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
@app.get("/")
async def root():
    return {
        "message": "Brain Tumor Classification API",
        "status": "running",
        "model_loaded": "model" in app_state,
        "device": str(DEVICE)
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in app_state
    }


# ============================================================
# EVALUATION ENDPOINT (with ground truth comparison)
# ============================================================
@app.post("/evaluate_tumor/")
async def evaluate_tumor_with_ground_truth(request: PredictionRequest):
    """
    Evaluate model predictions against ground truth labels.
    Useful for validation/testing with labeled data.
    """
    try:
        patient_path = request.patient_folder_path
        model_name = request.model_name
        
        # Convert to absolute path if relative
        if not os.path.isabs(patient_path):
            patient_path = os.path.abspath(patient_path)
            print(f"Converted relative path to absolute: {patient_path}")
        
        # Check if requested model exists
        if model_name not in app_state["models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(app_state['models'].keys())}"
            )
        
        if not os.path.exists(patient_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Path not found: {patient_path}"
            )
        
        # Get the requested model
        model = app_state["models"][model_name]
        
        # Load with ground truth
        print(f"Loading patient data with ground truth from: {patient_path}")
        img_tensor, mask_gt, grade_gt = preprocess_single_patient(
            patient_path, 
            return_ground_truth=True
        )
        img_tensor = img_tensor.to(app_state["device"])
        mask_gt = mask_gt.to(app_state["device"])
        
        # Run inference
        print(f"Running inference with model '{model_name}'...")
        with torch.no_grad():
            seg_logits, class_logits = model(img_tensor)
        
        # Classification results
        grade_prob = torch.sigmoid(class_logits).item()
        pred_diagnosis = "HGG" if grade_prob > 0.5 else "LGG"
        true_diagnosis = "HGG" if grade_gt.item() == 1 else "LGG"
        classification_correct = (pred_diagnosis == true_diagnosis)
        
        # Segmentation results
        pred_mask = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
        
        # Calculate Dice Score
        pred_binary = (pred_mask > 0).float()
        gt_binary = (mask_gt > 0).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        dice_score = (2.0 * intersection / (union + 1e-6)).item()
        
        # Visualization with ground truth
        pred_mask_np = pred_mask.cpu().numpy()[0]
        mask_gt_np = mask_gt.cpu().numpy()[0]
        input_t1_slice = img_tensor.cpu().numpy()[0, 1, :, :, 64]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MRI
        axes[0].imshow(input_t1_slice, cmap='gray')
        axes[0].set_title('MRI (T1 Slice)')
        axes[0].axis('off')
        
        # Ground Truth Mask
        cmap = plt.get_cmap('jet')
        axes[1].imshow(mask_gt_np[:, :, 64], cmap=cmap, vmin=0, vmax=3)
        axes[1].set_title(f'Ground Truth: {true_diagnosis}')
        axes[1].axis('off')
        
        # Predicted Mask
        axes[2].imshow(pred_mask_np[:, :, 64], cmap=cmap, vmin=0, vmax=3)
        axes[2].set_title(f'Prediction: {pred_diagnosis}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "model_used": model_name,
            "prediction": {
                "diagnosis": pred_diagnosis,
                "hgg_probability": round(grade_prob, 4),
                "lgg_probability": round(1 - grade_prob, 4)
            },
            "ground_truth": {
                "diagnosis": true_diagnosis,
                "grade_value": grade_gt.item()
            },
            "metrics": {
                "classification_correct": classification_correct,
                "dice_score": round(dice_score, 4)
            },
            "comparison_image_b64": img_base64,
            "message": "Evaluation complete with ground truth comparison."
        }
        
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"File not found: {str(e)}"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }