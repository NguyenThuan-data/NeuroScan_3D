"""
Brain Tumor Classification & Segmentation - Streamlit UI

This Streamlit app provides an interactive interface for:
- Uploading brain MRI scans (ZIP file with 4 modalities)
- Running tumor classification and segmentation
- Visualizing results with interactive charts
"""

import streamlit as st
import requests
import zipfile
import tempfile
import os
import base64
from PIL import Image
import io
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
# Use environment variable for API URL (for production deployment)
# Falls back to localhost for local development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict_tumor/"

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def validate_nifti_files(file_list):
    """
    Validate that the uploaded ZIP contains all required NIfTI files.
    
    Args:
        file_list: List of file names in the ZIP
    
    Returns:
        tuple: (is_valid, found_modalities, missing_modalities)
    """
    # Check more specific modalities first (t1ce before t1) to avoid substring matching issues
    required_modalities = ['flair', 't1ce', 't1', 't2']
    found_modalities = []
    
    for file_name in file_list:
        # Check if it's a NIfTI file
        if file_name.endswith(('.nii', '.nii.gz')):
            # Check which modality it contains
            file_lower = file_name.lower()
            for modality in required_modalities:
                if modality in file_lower:
                    found_modalities.append(modality)
                    break
    
    # Remove duplicates
    found_modalities = list(set(found_modalities))
    missing_modalities = [m for m in required_modalities if m not in found_modalities]
    
    is_valid = len(missing_modalities) == 0
    
    return is_valid, found_modalities, missing_modalities


def extract_zip_to_temp(uploaded_file):
    """
    Extract uploaded ZIP file to a temporary directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        str: Path to the temporary directory containing extracted files
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save uploaded file temporarily
    temp_zip_path = os.path.join(temp_dir, "upload.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract ZIP contents
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find the actual folder containing the NIfTI files
    # (in case the ZIP has a subfolder structure)
    for root, dirs, files in os.walk(extract_dir):
        nifti_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
        if len(nifti_files) >= 4:
            return root
    
    return extract_dir


def call_prediction_api(uploaded_file, model_name="default"):
    """
    Call the FastAPI prediction endpoint with ZIP file upload.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (ZIP file)
        model_name: Model to use for prediction
    
    Returns:
        dict: API response
    """
    try:
        # Prepare files and data for multipart form upload
        files = {
            'zip_file': (uploaded_file.name, uploaded_file.getvalue(), 'application/zip')
        }
        data = {
            'model_name': model_name
        }
        
        # Send POST request with file upload
        response = requests.post(PREDICT_ENDPOINT, files=files, data=data, timeout=300)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server.")
        st.warning("""
        **Possible reasons:**
        - API server is starting up (cold start on free tier ~30-60 seconds)
        - API server is not running
        - Incorrect API URL configured
        
        **If running locally:** Start the API with:
        ```bash
        uvicorn tumor_vision_api.api_logic:app --reload
        ```
        """)
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. The prediction is taking too long.")
        st.info("The model inference can take 2-5 minutes depending on server resources. Please wait...")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {str(e)}")
        return None


def decode_base64_image(base64_string):
    """
    Decode base64 image string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image
    
    Returns:
        PIL.Image: Decoded image
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


def create_probability_chart(hgg_prob, lgg_prob):
    """
    Create an interactive bar chart showing HGG vs LGG probabilities.
    
    Args:
        hgg_prob: HGG probability (0-1)
        lgg_prob: LGG probability (0-1)
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=['HGG (High-Grade)', 'LGG (Low-Grade)'],
            y=[hgg_prob * 100, lgg_prob * 100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{hgg_prob*100:.2f}%', f'{lgg_prob*100:.2f}%'],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Tumor Grade Probability Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=12), gridcolor='lightgray')
    
    return fig


# Main App
def main():
    # Header
    st.title("🧠 Brain Tumor Classification & Segmentation")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **Required Files:**
        - flair.nii (.gz)
        - t1.nii (.gz)
        - t1ce.nii (.gz)
        - t2.nii (.gz)
        
        **Classification:**
        - **HGG**: High-Grade Glioma
        - **LGG**: Low-Grade Glioma
        
        **Segmentation Classes:**
        - **NCR/NET**: Necrotic Core
        - **ED**: Edema
        - **ET**: Enhancing Tumor
        """)
        
        st.markdown("---")
        st.markdown("**API Status**")
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if health_response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Offline")
    
    # Main content
    st.header("📁 Upload Patient Data")
    st.markdown("Upload a ZIP file containing the 4 required NIfTI files (flair, t1, t1ce, t2)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Drag and drop ZIP file here or click to browse",
        type=['zip'],
        help="ZIP file should contain 4 NIfTI files: flair, t1, t1ce, t2"
    )
    
    # Initialize session state
    if 'uploaded_zip' not in st.session_state:
        st.session_state.uploaded_zip = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner("📦 Validating ZIP file..."):
            try:
                # Validate ZIP file (quick check without full extraction)
                zip_bytes = uploaded_file.getvalue()
                files_in_zip = []
                
                # Read ZIP file contents to validate
                with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
                    files_in_zip = zip_ref.namelist()
                
                # Validate files
                is_valid, found_modalities, missing_modalities = validate_nifti_files(files_in_zip)
                
                if is_valid:
                    st.session_state.uploaded_zip = uploaded_file
                    st.success(f"✅ ZIP file validated successfully!")
                    st.info(f"📋 Found modalities: **{', '.join(sorted(found_modalities))}**")
                else:
                    st.error(f"❌ Missing required modalities: **{', '.join(missing_modalities)}**")
                    st.warning("Please ensure your ZIP file contains all 4 modalities: flair, t1, t1ce, t2")
                    st.session_state.uploaded_zip = None
                    
            except zipfile.BadZipFile:
                st.error("❌ Invalid ZIP file. Please upload a valid ZIP archive.")
                st.session_state.uploaded_zip = None
            except Exception as e:
                st.error(f"❌ Error processing ZIP file: {str(e)}")
                st.session_state.uploaded_zip = None
    
    # Predict button
    st.markdown("---")
    predict_button = st.button(
        "🔍 Predict Tumor",
        type="primary",
        disabled=(st.session_state.uploaded_zip is None),
        use_container_width=True
    )
    
    if predict_button and st.session_state.uploaded_zip:
        with st.spinner("🔄 Running prediction... This may take a few moments."):
            result = call_prediction_api(st.session_state.uploaded_zip)
            
            if result and result.get("status") == "success":
                st.session_state.prediction_result = result
                st.success(f"✅ Prediction completed! Diagnosis: **{result['diagnosis']}**")
            elif result and result.get("status") == "error":
                st.error(f"❌ Prediction failed: {result.get('message', 'Unknown error')}")
            else:
                st.error("❌ Failed to get prediction from API")
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.header("📊 Classification Results")
        
        # Display diagnosis in a prominent box
        diagnosis = result['diagnosis']
        confidence = result['hgg_probability'] if diagnosis == 'HGG' else result['lgg_probability']
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class='success-box' style='text-align: center;'>
                <h2 style='margin: 0; color: #155724;'>Diagnosis: {diagnosis}</h2>
                <p style='font-size: 18px; margin: 10px 0 0 0;'>Confidence: {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability distribution chart
        st.markdown("### 📈 Probability Distribution")
        fig = create_probability_chart(result['hgg_probability'], result['lgg_probability'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Segmentation visualization
        st.markdown("---")
        st.header("🎨 Tumor Segmentation Visualization")
        
        try:
            # Decode and display the 4-panel segmentation image
            seg_image = decode_base64_image(result['predicted_mask_image_b64'])
            
            st.image(
                seg_image,
                caption="Tumor Segmentation Results: MRI | NCR/NET (Necrotic Core) | ED (Edema) | ET (Enhancing Tumor)",
                use_column_width=True
            )
            
            # Additional info
            st.info(f"🤖 Model used: **{result.get('model_used', 'default')}**")
            
        except Exception as e:
            st.error(f"❌ Error displaying segmentation image: {str(e)}")
        
        # Download results option
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Analyze Another Case", use_container_width=True):
                st.session_state.prediction_result = None
                st.session_state.uploaded_zip = None
                st.rerun()


if __name__ == "__main__":
    main()

