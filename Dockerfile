# Use Python 3.10 (critical for some packages like torch)
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Download dlib predictor if not already present
RUN if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then \
    wget -O shape_predictor_68_face_landmarks.dat.bz2 "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# Expose FastAPI default port
EXPOSE 8000

# Add healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
