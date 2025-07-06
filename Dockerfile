# Use Python 3.10 slim to match runtime.txt
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for dlib, OpenCV, moviepy, ImageMagick
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    bzip2 \
    cmake \
 && sed -i 's/rights="none"/rights="read|write"/g' /etc/ImageMagick-6/policy.xml \
 && rm -rf /var/lib/apt/lists/*


# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Download dlib predictor if missing
RUN if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# Use env PORT for Railway
ENV PORT 8080

# Expose port
EXPOSE ${PORT}

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
 CMD curl --fail http://localhost:${PORT}/healthz || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
