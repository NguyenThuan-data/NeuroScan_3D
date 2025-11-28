# Dockerfile for Hugging Face Spaces Deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY tumor_vision_api/ ./tumor_vision_api/
COPY models/ ./models/

# Expose HF Spaces standard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

# Run the application on HF Spaces port
CMD ["uvicorn", "tumor_vision_api.api_logic:app", "--host", "0.0.0.0", "--port", "7860"]

