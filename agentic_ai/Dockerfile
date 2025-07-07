# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV GEOCLIP_API_URL=http://geoclipapi:8090/predict
ENV LLAMA_API_URL=http://llamaapi:8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (excluding unnecessary directories)
COPY app.py .
COPY start.py .
COPY run_app.py .
COPY __init__.py .

# Copy directories
COPY mcp_server/ ./mcp_server/
COPY tools/ ./tools/
COPY agents/ ./agents/
COPY workflows/ ./workflows/

# Create logs directory (empty, will be mounted as volume)
RUN mkdir -p logs

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose the port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8501/ || exit 1

# Run the application
CMD ["python", "start.py"] 