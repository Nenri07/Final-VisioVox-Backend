from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import logging
from gtts import gTTS
import uuid
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_audioclips
import time
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lipreading API", version="1.0.0")

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("pretrain", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "port": os.getenv("PORT", "8000"),
        }
    )

@app.get("/healthz")
async def health_check_alt():
    """Alternative health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.get("/")
async def root():
    return {"message": "Lipreading API is running", "version": "1.0.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    unique_id = str(uuid.uuid4())
    
    # Output files
    audio_filename = f"audio_{unique_id}.mp3"
    audio_path = f"outputs/{audio_filename}"
    
    try:
        # For now, just return a simple prediction without complex processing
        prediction = "HELLO WORLD"
        
        # Generate audio file with gTTS
        try:
            tts = gTTS(text=str(prediction), lang='en', slow=False)
            tts.save(audio_path)
            logger.info(f"Generated audio file: {audio_path}")
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

        # Railway URL
        base_url = os.environ.get("RAILWAY_PUBLIC_DOMAIN", f"https://{os.environ.get('RAILWAY_STATIC_URL', 'localhost:8000')}")
        if not base_url.startswith('http'):
            base_url = f"https://{base_url}"
        
        # Return response
        audio_uri = f"{base_url}/outputs/{audio_filename}" if os.path.exists(audio_path) else None

        return JSONResponse(content={
            "prediction": str(prediction),
            "audioUri": audio_uri,
            "videoUri": None,
            "success": True
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename.endswith('.mp3'):
        return FileResponse(
            file_path,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif filename.endswith('.mp4'):
        return FileResponse(
            file_path,
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
