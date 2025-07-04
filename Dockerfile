# Stage 1: Builder stage for compiling dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (with CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --user --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir -r requirements.txt

# Download and extract dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor.dat.bz2 && \
    rm /models/shape_predictor.dat.bz2

# Stage 2: Final lightweight image
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /models /models

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat

# Create required directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Clean up apt cache
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose port (Railway will override $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/healthz || exit 1

# Start command (Railway will provide $PORT)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
