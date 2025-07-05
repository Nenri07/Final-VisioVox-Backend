FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies needed for compilation (dlib) and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    build-essential \ # Add build-essential for g++ and make
    curl \            # curl for healthcheck
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib predictor
RUN mkdir -p /models && \
    wget -q -O /models/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /models/shape_predictor_68_face_landmarks.dat.bz2

# Copy application code
COPY . .

# Set environment variables
ENV DLIB_SHAPE_PREDICTOR=/models/shape_predictor_68_face_landmarks.dat
ENV PYTHONUNBUFFERED=1 # Ensures Python output is unbuffered

# Create necessary directories
RUN mkdir -p static uploads outputs temp pretrain samples logs # Added samples and logs based on previous Dockerfile

# EXPOSE the default port for the application (Railway will map an external port to this)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1
    # Note: Using a fixed port for healthcheck if $PORT is giving issues for curl.
    # Railway's health check often runs *inside* the container, so it can hit localhost:8000
    # if your app correctly binds to 8000 (which ${PORT:-8000} handles).
    # You can change to `http://localhost:${PORT}/healthz` if you verify it works.

# Use ENTRYPOINT to ensure the shell processes the environment variable
# The ENTRYPOINT defines the primary command that will be executed when the container starts.
# CMD provides default arguments to the ENTRYPOINT.
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0"]

# CMD provides the default port. Railway injects the PORT env var.
# If PORT is set by Railway, it will be used. Otherwise, it defaults to 8000.
CMD ["--port", "${PORT:-8000}"]
