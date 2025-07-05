FROM python:3.10-slim

WORKDIR /app

# Install ONLY what we absolutely need
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p static uploads outputs temp pretrain models

# Start
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
