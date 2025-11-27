"""
Frontend Application for NeuroScan-3D-RAG
Supports both Streamlit and React frameworks
"""

import streamlit as st
import yaml
from pathlib import Path

# Load configuration
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def main():
    st.set_page_config(
        page_title="NeuroScan-3D-RAG",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 NeuroScan-3D-RAG")
    st.markdown("### Advanced 3D Medical Imaging Analysis with RAG")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Framework: {config['frontend']['framework']}")
        st.info(f"Device: {config['processing']['device']}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Upload DICOM", "Analysis Results", "Knowledge Base"])
    
    with tab1:
        st.header("Upload DICOM Files")
        uploaded_files = st.file_uploader(
            "Select DICOM files",
            type=['dcm', 'dicom'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            if st.button("Start Analysis"):
                st.info("Analysis in progress...")
                # TODO: Integrate with inference.py
    
    with tab2:
        st.header("Segmentation Results")
        st.info("Results will appear here after analysis")
        # TODO: Display 3D visualization and segmentation masks
    
    with tab3:
        st.header("Hospital Knowledge Base")
        st.info("Retrieved protocols and case studies")
        # TODO: Display RAG retrieval results

if __name__ == "__main__":
    main()

