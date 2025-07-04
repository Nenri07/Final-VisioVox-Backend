# Stage 1: Builder stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies including CMake
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU-only PyTorch
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir -r requirements.txt

# Download dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor.dat.bz2 && \
    rm /models/shape_predictor.dat.bz2

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app

# Copy only what we need from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /models /models
COPY . .

# Set environment
ENV PATH=/root/.local/bin:$PATH \
    DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat \
    PORT=8000

# Create needed directories
RUN mkdir -p static uploads

# Final cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/healthz || exit 1

# Run with Railway's PORT support
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
