FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
RUN mkdir -p static uploads outputs temp pretrain models

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
