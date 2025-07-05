FROM python:3.10-slim

WORKDIR /app

# Install only essential packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    ffmpeg \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib model
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Copy app
COPY . .

# Environment variables
ENV DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p static uploads outputs temp pretrain

# Start command - FIXED
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
