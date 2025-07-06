from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
from gtts import gTTS
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lipreading API", version="1.0.0")

# Create directories
os.makedirs("outputs", exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    return {"message": "Lipreading API is running", "version": "1.0.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    audio_filename = f"audio_{unique_id}.mp3"
    audio_path = f"outputs/{audio_filename}"
    
    try:
        prediction = "HELLO WORLD"
        tts = gTTS(text=prediction, lang='en', slow=False)
        tts.save(audio_path)
        
        return {
            "prediction": prediction,
            "audioUri": f"/outputs/{audio_filename}",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
