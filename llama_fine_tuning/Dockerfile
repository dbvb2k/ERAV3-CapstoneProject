# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Alternative base images if the above doesn't work:
# FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

# Alternative: Use Python 3.8 (available by default in Ubuntu 20.04)
# Uncomment the lines below and comment out the Python 3.10 installation if needed
# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
# RUN apt-get update && apt-get install -y \
#     python3.8 \
#     python3.8-dev \
#     python3-pip \
#     git wget curl build-essential \
#     && rm -rf /var/lib/apt/lists/*
# RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV IS_CONTAINER=true

# Set default environment variables (can be overridden)
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV LOG_LEVEL=INFO
ENV CORS_ORIGINS=*
ENV CORS_METHODS=*
ENV CORS_HEADERS=*

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install bitsandbytes with GPU support
RUN pip install bitsandbytes --upgrade --index-url https://jllllll.github.io/bitsandbytes-windows-webui

# Copy the application code
COPY llama_api.py .
COPY model/ ./model/

# Create logs directory
RUN mkdir -p logs

# Set CUDA environment variables for optimal performance
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create startup script with proper Unix line endings
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'set -e' >> /start.sh && \
    echo '' >> /start.sh && \
    echo 'echo "Setting up Llama API environment..."' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Check if GPU is available' >> /start.sh && \
    echo 'if command -v nvidia-smi &> /dev/null; then' >> /start.sh && \
    echo '    echo "NVIDIA GPU detected:"' >> /start.sh && \
    echo '    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits' >> /start.sh && \
    echo 'else' >> /start.sh && \
    echo '    echo "Warning: nvidia-smi not found. GPU may not be available."' >> /start.sh && \
    echo '    echo "Starting in CPU-only mode..."' >> /start.sh && \
    echo 'fi' >> /start.sh && \
    echo '' >> /start.sh && \
    echo 'echo "Starting Llama API server..."' >> /start.sh && \
    echo 'exec uvicorn llama_api:app --host 0.0.0.0 --port 8080 --workers 1 --log-level info' >> /start.sh

RUN chmod +x /start.sh

# Expose the port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use the startup script
CMD ["/start.sh"] 