# Alternative Debian-based Dockerfile (more compatible but larger)
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libgl1-mesa-dev \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    bzip2 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU version first
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Final runtime image
FROM python:3.10-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    libopenblas0 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libwebp6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    curl \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dlib model
COPY --from=builder /models /models

# Copy application code
COPY . .

# Set environment variables
ENV DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create required directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port
EXPOSE ${PORT:-8000}

# Start command
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
