# Stage 1: Builder stage for compiling dependencies
FROM python:3.10-slim-buster as builder

WORKDIR /app

# Install build-time system dependencies (including g++ and make for dlib/cmake)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    cmake \
    build-essential \
    ffmpeg \           # ffmpeg needed for moviepy compilation if any
    libsm6 \           # Moviepy dependency often needed
    libxext6 \         # Moviepy dependency often needed
    libxrender-dev \   # Moviepy dependency often needed
    libglib2.0-0 \     # Moviepy dependency often needed
    libgl1-mesa-glx \  # Moviepy dependency often needed for rendering
    libgomp1 \         # OpenMP dependency
    curl \             # For healthcheck in final stage
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (with CPU-only PyTorch, ensure dlib builds here)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download and extract dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Stage 2: Final lightweight image
FROM python:3.10-slim-buster

WORKDIR /app

# Copy only necessary files from builder
# Copy Python site-packages (where pip installed everything)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy dlib model
COPY --from=builder /models /models

# Copy application code
COPY . .

# Set environment variables
ENV DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat
ENV PYTHONUNBUFFERED=1

# Create required directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Clean up apt cache (important for final image size if any new apt packages were installed in this stage)
# In this specific multi-stage setup, almost everything is from the builder, but it's good practice.
RUN apt-get update && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose port (Railway will override $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Start command
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0"]
CMD ["--port", "${PORT:-8000}"]
