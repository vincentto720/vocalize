from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
import torch
from pathlib import Path
import uuid

app = FastAPI()

# CORS middleware to allow React Native app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Demucs model (htdemucs is the default high-quality model)
model = get_model(name='htdemucs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@app.get("/")
async def root():
    return {"message": "Vocalize API is running"}

@app.post("/api/separate")
async def separate_vocals(file: UploadFile = File(...)):
    """
    Receives an MP3 file, separates vocals using Demucs,
    returns the vocals track
    """
    if not file.filename.endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")
    
    # Generate unique ID for this processing job
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load audio file
        wav = AudioFile(str(upload_path)).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels
        )
        
        # Convert to tensor and add batch dimension
        wav = torch.from_numpy(wav).to(device)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        # Apply Demucs model
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device)[0]
        
        sources = sources * ref.std() + ref.mean()
        
        # Demucs outputs: drums, bass, other, vocals (index 3)
        vocals = sources[3].cpu().numpy()
        
        # Save vocals
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        vocals_file = job_output_dir / "vocals.wav"
        
        save_audio(vocals, vocals_file, samplerate=model.samplerate)
        
        if not vocals_file.exists():
            raise HTTPException(status_code=500, detail="Vocal separation failed")
        
        return {
            "job_id": job_id,
            "message": "Separation complete",
            "vocals_url": f"/api/download/{job_id}"
        }
    
    except Exception as e:
        # Cleanup on error
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/download/{job_id}")
async def download_vocals(job_id: str):
    """
    Downloads the separated vocals file
    """
    vocals_file = OUTPUT_DIR / job_id / "vocals.wav"
    
    if not vocals_file.exists():
        raise HTTPException(status_code=404, detail="Vocals file not found")
    
    return FileResponse(
        path=vocals_file,
        media_type="audio/wav",
        filename="vocals.wav"
    )

@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """
    Cleans up processed files
    """
    job_output_dir = OUTPUT_DIR / job_id
    
    if job_output_dir.exists():
        shutil.rmtree(job_output_dir)
    
    # Also clean up original upload
    for upload_file in UPLOAD_DIR.glob(f"{job_id}_*"):
        upload_file.unlink()
    
    return {"message": "Cleanup complete"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1738)