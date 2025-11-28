# NeuroScan Deployment Guide

This guide walks you through deploying the NeuroScan brain tumor classification application using:
- **Render.com** (Free tier) for the FastAPI backend
- **Streamlit Cloud** (Free tier) for the UI frontend

## Architecture Overview

```
User Browser
    ↓
Streamlit Cloud (UI)
    ↓ HTTPS API Calls
Render.com (FastAPI + ML Model)
```

---

## Prerequisites

- [x] GitHub account
- [x] Render.com account (sign up at https://render.com)
- [x] Streamlit Cloud account (sign up at https://share.streamlit.io)
- [x] All code changes committed to GitHub

---

## Part 1: Commit and Push to GitHub

### Step 1: Commit Your Changes

```bash
cd /c/Users/Admin1/OneDrive/Project/NeuroScan

# Review staged files
git status

# Commit the deployment configuration
git commit -m "Add deployment configuration for Render and Streamlit Cloud"
```

### Step 2: Push to GitHub

If this is your first push or you need to create a new repository:

```bash
# Create a new repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/NeuroScan.git
git push -u origin feature/tumor_api
```

Or if you already have a remote:

```bash
git push origin feature/tumor_api
```

**Note:** You may want to merge this into your main branch before deploying.

---

## Part 2: Deploy FastAPI Backend to Render

### Step 1: Choose Deployment Method

**Option A: Blueprint (Recommended - Automatic)**

1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub account and select **NeuroScan** repository
4. Render will auto-detect `render.yaml` and configure everything
5. Click **"Apply"** and skip to Step 3

**Option B: Manual Web Service (More Control)**

1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if you haven't already
4. Select the **NeuroScan** repository
5. Continue to Step 2

### Step 2: Configure the Web Service (Manual Method Only)

**IMPORTANT:** Make sure to select **Docker** as the environment!

Fill in the following settings:

| Setting | Value |
|---------|-------|
| **Name** | `neuroscan-api` (or your preferred name) |
| **Region** | Choose closest to you |
| **Branch** | `feature/tumor_api` (or `main`) |
| **Root Directory** | Leave blank (root) |
| **Environment** | **Docker** ⚠️ CRITICAL! |
| **Dockerfile Path** | `./Dockerfile` |
| **Docker Context** | `.` |
| **Docker Command** | Leave blank |
| **Plan** | **Free** |

### Step 3: Environment Variables

Render should auto-detect from `render.yaml`, but verify these are set:

- `PORT` = `8000`
- `PYTHON_VERSION` = `3.10`

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Wait for the build to complete (5-10 minutes first time)
3. Once deployed, you'll get a URL like: `https://neuroscan-api.onrender.com`
4. Test the health endpoint: `https://neuroscan-api.onrender.com/health`

**Important:** Save this URL! You'll need it for Streamlit configuration.

### Expected Output

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

## Part 3: Deploy Streamlit UI to Streamlit Cloud

### Step 1: Go to Streamlit Cloud

1. Visit https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**

### Step 2: Configure Deployment

| Setting | Value |
|---------|-------|
| **Repository** | `YOUR_USERNAME/NeuroScan` |
| **Branch** | `feature/tumor_api` (or `main`) |
| **Main file path** | `streamlit_app.py` |
| **App URL** (optional) | Choose a custom subdomain |

### Step 3: Advanced Settings

Click **"Advanced settings"** and configure:

#### Python Version
- Set to: `3.10` or `3.11`

#### Requirements File
- Use: `requirements_streamlit.txt`

### Step 4: Configure Secrets

This is **CRITICAL** - click on **"Advanced settings"** → **"Secrets"**

Add the following in TOML format:

```toml
API_BASE_URL = "https://neuroscan-api.onrender.com"
```

**Replace** `neuroscan-api.onrender.com` with your actual Render URL from Part 2!

### Step 5: Deploy

1. Click **"Deploy!"**
2. Wait 2-3 minutes for deployment
3. Your app will be available at: `https://YOUR-APP-NAME.streamlit.app`

---

## Part 4: Testing End-to-End

### Step 1: Access Your Streamlit App

Visit your Streamlit Cloud URL (e.g., `https://neuroscan.streamlit.app`)

### Step 2: Check API Status

Look at the **sidebar** - it should show:
- ✅ **API Connected** (green)

If you see ❌ **API Offline** (red):
- Wait 30-60 seconds (cold start on free tier)
- Refresh the page
- Check that the `API_BASE_URL` secret is correct

### Step 3: Upload Test Data

1. Prepare a ZIP file with 4 NIfTI files (flair, t1, t1ce, t2)
2. Upload the ZIP file
3. Click **"Predict Tumor"**
4. Wait for results (may take 2-5 minutes on free tier)

### Expected Flow

```
Upload ZIP → Extract files → Call API → Wait for wake-up (if cold)
→ Run ML inference → Display results
```

---

## Important Notes for Free Tier Users

### Cold Starts on Render

- ⏰ **15-minute inactivity timeout** - API sleeps after 15 min of no requests
- 🚀 **Wake-up time** - First request takes 30-60 seconds
- ⚡ **Subsequent requests** - Fast (< 1 second) after wake-up

**Workaround:** Have Streamlit ping `/health` endpoint every 10 minutes to keep API awake.

### Resource Limits

**Render Free Tier:**
- 512 MB RAM (sufficient for inference on CPU)
- 750 hours/month (should be enough)
- Shared CPU (inference takes 1-3 minutes)

**Streamlit Cloud Free Tier:**
- 1 GB RAM
- 1 CPU core
- Unlimited apps (public repos)

### CORS Configuration

Already configured in `tumor_vision_api/api_logic.py`:
- ✅ Allows requests from `*.streamlit.app`
- ✅ Allows all HTTP methods
- ✅ Handles credentials properly

---

## Troubleshooting

### Issue: "Cannot connect to API server"

**Solution:**
1. Check that Render deployment is successful
2. Verify API URL in Streamlit secrets
3. Wait 60 seconds for cold start
4. Check Render logs for errors

### Issue: "Request timeout"

**Solution:**
- Normal on first cold start (can take 60s)
- Subsequent requests should be faster
- Consider upgrading to paid tier if this persists

### Issue: "CORS error"

**Solution:**
- Verify CORS middleware is configured in `api_logic.py`
- Check that Streamlit app URL matches CORS allowed origins
- Try adding specific Streamlit URL to allowed origins

### Issue: "Model not found" on Render

**Solution:**
- Verify `models/best_model.pth` is in your Git repository
- Check that `.gitignore` doesn't exclude model files (line 67 should be commented)
- Review Render build logs

---

## Monitoring and Logs

### Render Logs

View real-time logs:
1. Go to Render dashboard
2. Select your web service
3. Click **"Logs"** tab
4. Watch for startup messages and errors

### Streamlit Logs

View app logs:
1. Go to Streamlit Cloud dashboard
2. Select your app
3. Click **"Manage app"** → **"Logs"**

---

## Updating Your Deployment

### Update Backend (Render)

```bash
# Make changes to api_logic.py or model_utils.py
git add .
git commit -m "Update API logic"
git push origin feature/tumor_api
```

Render will **auto-deploy** on push (takes 5-10 minutes).

### Update Frontend (Streamlit)

```bash
# Make changes to streamlit_app.py
git add .
git commit -m "Update UI"
git push origin feature/tumor_api
```

Streamlit Cloud will **auto-deploy** on push (takes 2-3 minutes).

---

## Upgrading to Paid Plans (Optional)

### When to Upgrade?

Consider paid plans if:
- Cold starts are unacceptable (30-60 sec delay)
- You need faster inference (GPU)
- You expect high traffic (> 750 hours/month)
- You need better reliability

### Render Pricing

- **Starter Plan:** $7/month
  - No cold starts
  - Always active
  - Same resources as free tier

- **Standard Plan:** $25/month
  - More RAM (2 GB)
  - Faster CPU
  - Better for production

### Streamlit Cloud Pricing

Free tier is usually sufficient for most use cases!

- **Team Plan:** $250/month (for organizations)

---

## Security Considerations

### API Authentication (Future Enhancement)

Current setup has **no authentication** on API. For production:

1. Add API key authentication
2. Store API key in Streamlit secrets
3. Validate API key on each request

Example:

```python
# In api_logic.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
```

### Data Privacy

- No patient data is stored on servers
- All processing happens in memory
- Temporary files are cleaned up after each request
- Consider HIPAA compliance for medical use

---

## Next Steps

✅ **You're deployed!** Your app is now live and accessible worldwide.

**Recommended actions:**
1. Share your Streamlit URL with users/stakeholders
2. Monitor logs for errors in first few days
3. Test with various MRI scan datasets
4. Consider adding authentication for production use
5. Set up monitoring/alerting (e.g., UptimeRobot)

---

## Support and Resources

- **Render Docs:** https://render.com/docs
- **Streamlit Docs:** https://docs.streamlit.io/streamlit-cloud
- **GitHub Repository:** https://github.com/YOUR_USERNAME/NeuroScan

---

## Local Development Setup (Optional)

If you want to run both components locally:

```bash
# Install backend dependencies
pip install -r requirements_api.txt

# Install frontend dependencies
pip install -r requirements_streamlit.txt

# Run API locally
uvicorn tumor_vision_api.api_logic:app --reload

# Run Streamlit locally (in another terminal)
streamlit run streamlit_app.py
```

## Summary of What You Created

```
NeuroScan/
├── Dockerfile                          # Backend container configuration
├── render.yaml                         # Render deployment config
├── requirements_api.txt                # Backend dependencies (Render)
├── requirements_streamlit.txt          # Frontend dependencies (Streamlit Cloud)
├── .streamlit/
│   ├── config.toml                    # Streamlit configuration
│   └── secrets.toml.example           # Example secrets file
├── streamlit_app.py                   # ✅ Updated with env vars
└── tumor_vision_api/
    └── api_logic.py                   # ✅ Added CORS middleware
```

**Note:** The original `requirements.txt` was removed to avoid confusion. We now use separate requirement files for each deployment target.

**All changes are staged and ready to commit!** 🚀

