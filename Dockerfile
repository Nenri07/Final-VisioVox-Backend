# Multi-stage build for Railway deployment
FROM python:3.10-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    cmake \
    openblas-dev \
    jpeg-dev \
    libpng-dev \
    tiff-dev \
    libwebp-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    mesa-dev \
    glib-dev \
    ffmpeg-dev \
    wget \
    bzip2 \
    git

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Final runtime image
FROM python:3.10-alpine AS runtime

# Install runtime dependencies
RUN apk add --no-cache \
    ffmpeg \
    libgomp \
    openblas \
    jpeg \
    libpng \
    tiff \
    libwebp \
    libx11 \
    libxext \
    libxrender \
    mesa \
    glib \
    curl \
    imagemagick

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
