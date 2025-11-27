import matplotlib.pyplot as plt
from PIL import Image   
import io
import base64 # For sending image data directly in JSON
import numpy as np
import sys
import os
from .data_preprocessing_utils import BratsDataset

# Add project root to Python path so imports work when running uvicorn
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tumor_vision_api.model_utils import MultiTaskUNet, HybridLoss
from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app_state = {} # Global dictionary to store model reference

# The lifespan context loads the model on startup and unloads on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API STARTING: Loading model into memory...")
    
    # 1. Initialize the model structure
    model = MultiTaskUNet(in_channels=4, num_classes=4)
    
    # 2. Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Store the loaded model globally for endpoints to access
    app_state["model"] = model
    app_state["device"] = DEVICE
    
    print(f"Model loaded on: {DEVICE}")
    yield # API is ready to serve requests
    
    # API Shutdown logic (optional)
    app_state.clear()
    print("API SHUTDOWN: Model unloaded.")

app = FastAPI(lifespan=lifespan)

# ----------------------------------------------------
# C. THE PREDICTION ENDPOINT
# ----------------------------------------------------

@app.post("/predict_tumor/")
async def predict_tumor_analysis(nifti_path: str):
    try:
        # --- Existing Prediction Logic (from previous steps) ---
        img_tensor = torch.randn(1, 4, 128, 128, 128).to(app_state["device"]) # Placeholder
        with torch.no_grad():
            seg_logits, class_logits = app_state["model"](img_tensor)
        
        grade_prob = torch.sigmoid(class_logits).item()
        diagnosis = "HGG" if grade_prob > 0.5 else "LGG"
        
        # Get the predicted mask (as a NumPy array)
        # Note: argmax gives you the class with the highest probability
        pred_mask_np = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).cpu().numpy()[0]
        # And let's get one T1 slice from the input for background
        input_t1_slice = img_tensor.cpu().numpy()[0, 0, :, :, 64] # Example: T1 modality, mid-slice (adjust index)
        
        # --- NEW: VISUALIZATION LOGIC ---
        # 1. Create a figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Two plots: MRI and Pred Mask
        
        # Plot MRI (T1) background
        axes[0].imshow(input_t1_slice, cmap='gray')
        axes[0].set_title('MRI (T1 Slice)')
        axes[0].axis('off')

        # Plot predicted mask
        # Colors: 0=Background (blue), 1=NCR/NET (red), 2=ED (yellow), 3=ET (dark red)
        # You'll need to define a colormap for your 4 classes
        cmap = plt.get_cmap('jet') # jet has distinct colors
        cmap.set_bad('black') # for background
        
        # Make background transparent if needed, or use a specific color for 0
        # For display, often mask values 1,2,3 are used over a black background.
        # Here we'll just plot the mask values.
        # pred_mask_display = np.ma.masked_where(pred_mask_np == 0, pred_mask_np) # If you want to overlay
        
        axes[1].imshow(pred_mask_np[:, :, 64], cmap=cmap, vmin=0, vmax=3) # Mid-slice for mask
        axes[1].set_title(f'Pred: {diagnosis}')
        axes[1].axis('off')

        plt.tight_layout()
        
        # 2. Save the figure to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig) # Close the figure to free memory

        # 3. Encode the image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "diagnosis": diagnosis,
            "hgg_probability": round(grade_prob, 4),
            "predicted_mask_image_b64": img_base64, # <-- NEW: The image data
            "message": "Analysis complete with segmentation visualization."
        }

    except Exception as e:
        # ... (error handling) ...
        return {"status": "error", "message": f"An error occurred: {str(e)}"}