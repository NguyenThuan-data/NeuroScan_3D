---
title: NeuroScan API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# NeuroScan Brain Tumor Classification API

FastAPI backend for brain tumor classification and segmentation using Multi-Task U-Net.

## Overview

This API analyzes brain MRI scans to:
- **Classify** tumors as High-Grade Glioma (HGG) or Low-Grade Glioma (LGG)
- **Segment** tumor regions (Necrotic Core, Edema, Enhancing Tumor)

## API Endpoints

### Health Check
```
GET /health
```

Returns API status and model loading state.

### Predict Tumor
```
POST /predict_tumor/
```

**Request:**
- `zip_file`: ZIP file containing 4 NIfTI modalities (flair, t1, t1ce, t2)
- `model_name`: Model to use (default: "default")

**Response:**
```json
{
  "status": "success",
  "diagnosis": "HGG" or "LGG",
  "hgg_probability": 0.85,
  "lgg_probability": 0.15,
  "predicted_mask_image_b64": "base64_encoded_image",
  "model_used": "default"
}
```

## Usage

### cURL Example

```bash
curl -X POST "https://YOUR_USERNAME-neuroscan-api.hf.space/predict_tumor/" \
  -F "zip_file=@patient_scan.zip" \
  -F "model_name=default"
```

### Python Example

```python
import requests

url = "https://YOUR_USERNAME-neuroscan-api.hf.space/predict_tumor/"
files = {"zip_file": open("patient_scan.zip", "rb")}
data = {"model_name": "default"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Model

- **Architecture:** Multi-Task U-Net
- **Input:** 4 MRI modalities (FLAIR, T1, T1CE, T2)
- **Output:** Tumor classification + segmentation mask
- **Dataset:** BraTS 2020

## Hardware

- **CPU:** Intel/AMD (inference time: ~2-3 minutes)
- **RAM:** 16GB (sufficient for processing)

## License

Research and educational use only.

