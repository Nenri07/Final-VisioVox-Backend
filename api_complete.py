from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import logging
from gtts import gTTS
import uuid
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_lip_reading
import tempfile
from dotenv import load_dotenv
import gdown

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080")

# 🔧 Fixed: use only FILE ID (not full URL)
MODEL_CONFIG = {
    "weights": {
        "id": "10AgIULFG8Ic6mopDlJ-_BucGy1lEjQhr",  # ✅ your .pt file
        "path": "pretrain/LipCoordNet.pt",
        "expected_size": 25000000  # ~25MB
    },
    "predictor": {
        "id": "1M_hRmE2zezyQrf3XyUjc2gTOFXcGmbLL",  # ✅ shape_predictor file
        "path": "lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat",
        "expected_size": 63000000  # ~63MB
    }
}

# Create folders
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("pretrain", exist_ok=True)
os.makedirs("lip_coordinate_extraction", exist_ok=True)


def download_with_retry(file_id, destination, expected_size, max_retries=3):
    """Download a file from Google Drive using gdown with retries"""
    for attempt in range(max_retries):
        try:
            gdown.download(id=file_id, output=destination, quiet=False)
            size = os.path.getsize(destination)
            if size < expected_size * 0.9:
                raise ValueError(f"Downloaded file too small: {size} bytes")
            logger.info(f"✅ Downloaded {destination} ({size} bytes)")
            return True
        except Exception as e:
            logger.warning(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
    return False


def ensure_models():
    """Ensure both model weight and predictor are downloaded"""
    if not os.path.exists(MODEL_CONFIG["weights"]["path"]):
        logger.info("📥 Downloading model weights...")
        if not download_with_retry(
            MODEL_CONFIG["weights"]["id"],
            MODEL_CONFIG["weights"]["path"],
            MODEL_CONFIG["weights"]["expected_size"]
        ):
            raise RuntimeError("❌ Failed to download model weights")

    if not os.path.exists(MODEL_CONFIG["predictor"]["path"]):
        logger.info("📥 Downloading shape predictor...")
        if not download_with_retry(
            MODEL_CONFIG["predictor"]["id"],
            MODEL_CONFIG["predictor"]["path"],
            MODEL_CONFIG["predictor"]["expected_size"]
        ):
            raise RuntimeError("❌ Failed to download shape predictor")


# Load models at startup
try:
    ensure_models()
except Exception as e:
    logger.error(f"CRITICAL: Model initialization failed - {str(e)}")
    raise

# FastAPI app setup
app = FastAPI(
    title="Lipreading API",
    description="API for lip-reading from videos and generating audio output",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "weights": os.path.exists(MODEL_CONFIG["weights"]["path"]),
            "predictor": os.path.exists(MODEL_CONFIG["predictor"]["path"])
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, detail="Only MP4, MOV or AVI files allowed")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, detail="File too large (max 100MB)")

    unique_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()

    try:
        # Save the uploaded video
        video_path = os.path.join(temp_dir, f"input_{unique_id}.mp4")
        with open(video_path, "wb") as f:
            f.write(content)

        # Run the prediction
        prediction = predict_lip_reading(
            video_path=video_path,
            weights_path=MODEL_CONFIG["weights"]["path"],
            device="cpu"
        )

        # Generate audio from prediction
        audio_path = os.path.join("outputs", f"output_{unique_id}.mp3")
        tts = gTTS(text=prediction, lang='en')
        tts.save(audio_path)

        return {
            "prediction": prediction,
            "audio_url": f"{BASE_URL}/outputs/{os.path.basename(audio_path)}",
            "success": True
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
