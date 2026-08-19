FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU/CUDA dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and datasets
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
