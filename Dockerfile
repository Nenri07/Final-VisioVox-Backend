FROM python:3.10-slim-buster  # <--- CRITICAL CHANGE: Changed from 3.9 to 3.10

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
    bzip2 \ # Added bzip2 for bunzip2 command
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static uploads outputs temp pretrain samples logs

# Download dlib predictor if not present
# Corrected the wget URL and command to be more robust
RUN if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then \
    wget -O shape_predictor_68_face_landmarks.dat.bz2 "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# Expose port
# IMPORTANT: Railway often uses the PORT environment variable.
# Your FastAPI app should listen on the port specified by Railway's PORT env var,
# or default to 8000 (as per your config.py APP_PORT).
# If you explicitly expose 8080, make sure your app listens on it.
# It's better to expose the port your app actually uses, and your config.py defaults to 8000.
EXPOSE 8000 # Changed to 8000 to match your config.py default, or better, use $PORT in CMD

# Health check
# Ensure the healthcheck port matches your exposed port and app's listening port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1 # Changed to 8000

# Start command
# Your config.py implies your app listens on APP_PORT, which defaults to 8000.
# The `CMD` in the Dockerfile should align with how you start your FastAPI app.
# If `api.py` contains `uvicorn main:app`, then the command should be uvicorn.
# Assuming your main FastAPI app instance is called `app` in `main.py`
# If your main FastAPI app instance is called `app` in `api.py`, then use `api:app`.
# Use the $PORT environment variable provided by Railway.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
# Assuming your main FastAPI app is in `main.py` and the app instance is `app`.
# If it's in `api.py`, change `main:app` to `api:app`.
