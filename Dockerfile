# Dockerfile for FastAPI Backend
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY tumor_vision_api/ ./tumor_vision_api/
COPY models/ ./models/

# Expose port (7860 for Hugging Face Spaces, 8000 for Render)
# Use environment variable to support both platforms
EXPOSE 7860
EXPOSE 8000

# Health check (use PORT env var or default to 7860)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; import os; port = os.getenv('PORT', '7860'); requests.get(f'http://localhost:{port}/health')"

# Run the application on port from environment or default to 7860 (HF Spaces default)
CMD uvicorn tumor_vision_api.api_logic:app --host 0.0.0.0 --port ${PORT:-7860}

