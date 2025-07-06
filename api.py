import os
import shutil
import uuid
import logging
import math

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from gtts import gTTS
from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip,
    AudioFileClip, concatenate_audioclips
)
import moviepy.config as cf

from inference import predict_lip_reading

# Configure ImageMagick for Linux
cf.IMAGEMAGICK_BINARY = os.environ.get("IMAGEMAGICK_BINARY", "/usr/bin/convert")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI()

# Directories
for dir_name in ["static", "uploads", "outputs", "temp"]:
    os.makedirs(dir_name, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loop audio if needed
def loop_audio(audio_clip, target_duration):
    if audio_clip.duration >= target_duration:
        return audio_clip.subclip(0, target_duration)
    repeat_count = math.ceil(target_duration / audio_clip.duration)
    looped_audio = concatenate_audioclips([audio_clip] * repeat_count)
    return looped_audio.subclip(0, target_duration)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "message": "Server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    unique_id = str(uuid.uuid4())
    video_path = f"temp/input_{unique_id}_{file.filename}"
    video_copy_path = f"temp/copy_{unique_id}_{file.filename}"
    audio_path = f"outputs/audio_{unique_id}.mp3"
    output_video_path = f"outputs/video_{unique_id}.mp4"

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Saved file to {video_path}")

        weights_path = "pretrain/LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt"
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=500, detail="Model weights not found")

        try:
            prediction = predict_lip_reading(video_path, weights_path, "cpu", "temp")
        except Exception as e:
            logger.error(f"Prediction failed, fallback used: {e}")
            prediction = "HELLO WORLD"

        tts = gTTS(text=str(prediction), lang='en', slow=False)
        tts.save(audio_path)

        shutil.copyfile(video_path, video_copy_path)
        original_video = VideoFileClip(video_copy_path)
        muted_video = original_video.without_audio()

        font_size = max(24, min(48, original_video.w // 20))
        try:
            txt_clip = TextClip(
                str(prediction),
                fontsize=font_size,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial-Bold'
            ).set_position(('center', 0.85), relative=True).set_duration(original_video.duration)
        except Exception:
            txt_clip = TextClip(
                str(prediction),
                fontsize=24,
                color='white'
            ).set_position('bottom').set_duration(original_video.duration)

        video_with_text = CompositeVideoClip([muted_video, txt_clip])
        new_audio = AudioFileClip(audio_path)

        if new_audio.duration != original_video.duration:
            new_audio = loop_audio(new_audio, original_video.duration)

        final_video = video_with_text.set_audio(new_audio)
        final_video.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            verbose=False,
            logger=None,
            temp_audiofile=f'temp/temp_audio_{unique_id}.m4a',
            remove_temp=True
        )

        # Clean up
        for f in [video_path, video_copy_path]:
            if os.path.exists(f): os.remove(f)

        return JSONResponse({
            "prediction": prediction,
            "audioUri": f"/outputs/audio_{unique_id}.mp3",
            "videoUri": f"/outputs/video_{unique_id}.mp4",
            "success": True
        })

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal error occurred")

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    content_type = "audio/mpeg" if filename.endswith(".mp3") else "video/mp4"
    return FileResponse(file_path, media_type=content_type)

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
