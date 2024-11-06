# API Implementation

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
import torchaudio
import tempfile
import os
from datetime import datetime
import uuid
from pathlib import Path
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    save_spectrogram,
)
from transformers import pipeline
import soundfile as sf
import uvicorn
import io

app = FastAPI(
    title="Text-to-Speech API",
    description="API for voice cloning and text-to-speech synthesis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

# Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0

# Initializing models
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Pydantic models for request/response validation - Defining Schema
class TTSRequest(BaseModel):
    reference_text: Optional[str] = Field(None, description="Optional reference text for the audio")
    generate_text: str = Field(..., description="Text to be generated in the cloned voice")
    remove_silence: bool = Field(True, description="Whether to remove silence from the generated audio")

class TTSResponse(BaseModel):
    audio_url: str
    spectrogram_url: str
    duration: float
    sample_rate: int

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float

def load_model():
    """Initialize the TTS model"""
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=UNetT(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

model = load_model()

def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    """Clean up temporary files older than max_age_hours"""
    current_time = datetime.now()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                file_path.unlink()

@app.get("/")
def root():
    return {"data":"API's for Voice Cloning"}

# This API route is for transcription
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    if not audio_file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(400, "Unsupported audio format")
    
    temp_path = TEMP_DIR / f"{uuid.uuid4()}.wav"
    try:
        content = await audio_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        result = pipe(
            str(temp_path),
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )
        
        return TranscriptionResponse(
            text=result["text"].strip(),
            confidence=float(result.get("confidence", 1.0))
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

# This API route is for Synthesizing
@app.post("/api/synthesize", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    reference_audio: UploadFile = File(...)):
    
    """Synthesize speech using the reference audio and input text"""
    if not reference_audio.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(400, "Unsupported audio format")

    # Generate unique IDs for output files
    output_id = uuid.uuid4()
    audio_path = TEMP_DIR / f"output_{output_id}.wav"
    spectrogram_path = TEMP_DIR / f"spectrogram_{output_id}.png"

    try:
        content = await reference_audio.read()
        ref_audio_path = TEMP_DIR / f"ref_{output_id}.wav"
        with open(ref_audio_path, "wb") as f:
            f.write(content)

        # Process audio with pydub
        aseg = AudioSegment.from_file(ref_audio_path)
        if request.remove_silence:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
            )
            aseg = sum(non_silent_segs, AudioSegment.silent(duration=0))

        if len(aseg) > 15000:
            aseg = aseg[:15000]
        aseg.export(ref_audio_path, format="wav")

        # Getting reference text if not provided
        ref_text = request.reference_text
        if not ref_text:
            ref_text = pipe(
                str(ref_audio_path),
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()

        if not ref_text.endswith(". "):
            ref_text += ". " if not ref_text.endswith(".") else " "

        # Loading and processing audio
        audio, sr = torchaudio.load(ref_audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
            
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        
        audio = audio.to(device)

        # Generating speech
        text_list = [ref_text + request.generate_text]
        duration = audio.shape[-1] // hop_length + int(audio.shape[-1] / hop_length / len(ref_text) * len(request.generate_text) / speed)

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        # Post-process generated audio
        generated = generated.to(torch.float32)
        generated = generated[:, audio.shape[-1] // hop_length:, :]
        generated_mel_spec = generated.permute(0, 2, 1)
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        sf.write(audio_path, generated_wave, target_sample_rate)
        
        save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

        background_tasks.add_task(cleanup_old_files, TEMP_DIR)

        return TTSResponse(
            audio_url=f"/api/audio/{output_id}",
            spectrogram_url=f"/api/spectrogram/{output_id}",
            duration=float(len(generated_wave) / target_sample_rate),
            sample_rate=target_sample_rate
        )

    except Exception as e:
        raise HTTPException(500, f"Error during synthesis: {str(e)}")

# This API route will retrieve audio file
@app.get("/api/audio/{file_id}")
async def get_audio(file_id: str):
    """Retrieve generated audio file"""
    file_path = TEMP_DIR / f"output_{file_id}.wav"
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"generated_{file_id}.wav"
    )

# This API route is for getting spectogram
@app.get("/api/spectrogram/{file_id}")
async def get_spectrogram(file_id: str):
    """Retrieve generated spectrogram"""
    file_path = TEMP_DIR / f"spectrogram_{file_id}.png"
    if not file_path.exists():
        raise HTTPException(404, "Spectrogram not found")
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"spectrogram_{file_id}.png"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)