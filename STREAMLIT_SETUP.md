# Streamlit UI Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
# Activate your virtual environment
source venv_api/Scripts/activate  # Git Bash on Windows
# OR
venv_api\Scripts\activate.bat      # CMD on Windows

# Install new dependencies
pip install streamlit requests plotly
```

### 2. Start the FastAPI Server
```bash
# In terminal 1
cd C:\Users\Admin1\OneDrive\Project\NeuroScan
source venv_api/Scripts/activate
uvicorn tumor_vision_api.api_logic:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Start the Streamlit App
```bash
# In terminal 2 (new terminal)
cd C:\Users\Admin1\OneDrive\Project\NeuroScan
source venv_api/Scripts/activate
streamlit run streamlit_app.py
```

The Streamlit UI will open automatically at: http://localhost:8501

### 4. Test the Application

#### Prepare Test Data
1. Navigate to `data/nii_test_sample/` folder
2. Create a ZIP file containing the 4 NIfTI files:
   - BraTS20_Training_001_flair.nii
   - BraTS20_Training_001_t1.nii
   - BraTS20_Training_001_t1ce.nii
   - BraTS20_Training_001_t2.nii

**Windows (PowerShell)**:
```powershell
Compress-Archive -Path "data\nii_test_sample\*" -DestinationPath "test_patient.zip"
```

**Windows (Right-click)**:
- Right-click on the `nii_test_sample` folder
- Select "Send to" > "Compressed (zipped) folder"

#### Use the UI
1. Open http://localhost:8501 in your browser
2. Drag and drop `test_patient.zip` into the upload area
3. Wait for the success notification: "✅ Files uploaded successfully!"
4. Click the "🔍 Predict Tumor" button
5. Wait for prediction to complete (watch for notification)
6. View results:
   - Classification: HGG or LGG with confidence
   - Bar chart showing probabilities
   - 4-panel segmentation image

## Troubleshooting

### API Not Connected
If you see "❌ API Offline" in the sidebar:
1. Check that FastAPI is running: `curl http://localhost:8000/health`
2. Ensure uvicorn is running without errors
3. Check firewall settings if needed

### Upload Errors
If files are not recognized:
- Ensure ZIP contains exactly 4 .nii or .nii.gz files
- File names should contain: flair, t1, t1ce, t2
- Check that files are in the root of the ZIP (not in nested folders)

### Prediction Errors
If prediction fails:
- Check API terminal for error messages
- Verify model file exists: `models/best_model.pth`
- Ensure all 4 modality files were properly extracted

## Features

✅ Drag-and-drop ZIP file upload
✅ Automatic file validation
✅ Success notifications for upload and prediction
✅ Interactive probability bar chart
✅ Beautiful 4-panel tumor segmentation visualization
✅ Real-time API status indicator
✅ Responsive layout
✅ Error handling with user-friendly messages

## Architecture

```
┌─────────────────┐      HTTP Request       ┌──────────────────┐
│  Streamlit UI   │────────────────────────>│  FastAPI Server  │
│  (Port 8501)    │                         │  (Port 8000)     │
│                 │<────────────────────────│                  │
│  - Upload ZIP   │      JSON Response      │  - Load Model    │
│  - Display      │                         │  - Preprocess    │
│  - Charts       │                         │  - Predict       │
└─────────────────┘                         └──────────────────┘
```

