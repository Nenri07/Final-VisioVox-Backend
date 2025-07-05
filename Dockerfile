# Stage 1: Builder stage for compiling dependencies
FROM python:3.10-slim-buster as builder # Using -buster for broader apt-get compatibility

WORKDIR /app

# Install build-time system dependencies (including g++ and make for dlib/cmake)
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    bzip2 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (with CPU-only PyTorch, ensure dlib builds here)
# Using --break-system-packages for modern pip on Debian-based slim images if needed
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download and extract dlib model
# Ensure the downloaded file matches the expected name
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Stage 2: Final lightweight image
FROM python:3.10-slim-buster # Using -buster for consistency and smaller image

WORKDIR /app

# Copy only necessary files from builder (installed packages and dlib model)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# If pip install --user was used in builder, it would be /root/.local/lib/python3.10/site-packages,
# but the first stage RUN command uses default installation to /usr/local/lib
COPY --from=builder /models /models

# Copy application code
COPY . .

# Set environment variables for dlib model path
ENV DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat

# Create required directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Clean up apt cache (do this in the final stage to keep it small)
RUN apt-get update && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose port (Railway will override $PORT)
EXPOSE 8000

# Health check
# Use the $PORT variable provided by Railway for the healthcheck URL
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Start command (Railway will provide $PORT)
# Assuming your main FastAPI app is in `api.py` and the app instance is `app`.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
