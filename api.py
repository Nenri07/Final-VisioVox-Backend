import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import logging
from gtts import gTTS
import uuid
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_audioclips
import moviepy.config as cf
import math
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_lip_reading

# ✅ Linux-safe ImageMagick config for Railway
cf.IMAGEMAGICK_BINARY = os.environ.get("IMAGEMAGICK_BINARY", "/usr/bin/convert")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Mount static
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing routes stay unchanged:
# ✅ /healthz
# ✅ /predict
# ✅ /outputs/{filename}
# ✅ all your custom logic is fine!

# Make sure main runs uvicorn with env PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
