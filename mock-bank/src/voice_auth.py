# voice_auth_speechbrain.py
import os, csv, time
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import io
import uuid
import json
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# from audio_utils import read_wav_bytes_to_tensor, vad_trim, rms_normalize
from speechbrain.inference import EncoderClassifier

LOG_FILE = "score_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,"w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["ts","username","score","accepted","note"])
# ---------- CONFIG ----------
DATA_DIR = "./data"
PROFILE_FILENAME = "profile.npy"
EMB_LIST_FILENAME = "embeddings.npy"
METADATA_FILENAME = "meta.json"
SIMILARITY_THRESHOLD = 0.78   # tune this on real data
TARGET_SAMPLE_RATE = 16000
# ----------------------------




app = FastAPI(title="Local Voice-Auth (SpeechBrain, file-store)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the UI folder (adjust path if your web_ui is elsewhere)
ui_dir = os.path.join(os.path.dirname(__file__), ".", "web_ui")  # adjust relative path if needed
# or simply: ui_dir = "web_ui"
app.mount("/ui", StaticFiles(directory=ui_dir), name="static")
# Load SpeechBrain ECAPA model once (keeps in memory)
# This will download model weights the first time (saved under ./pretrained by default)
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


def ensure_user_dirs(username: str):
    user_dir = os.path.join(DATA_DIR, username)
    audio_dir = os.path.join(user_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    return user_dir, audio_dir


def read_audio_bytes_to_tensor(wav_bytes: bytes):
    """
    Read bytes (wav or similar decodable by soundfile) and return a mono torch.FloatTensor
    at TARGET_SAMPLE_RATE with shape (1, num_samples).
    """
    bio = io.BytesIO(wav_bytes)
    try:
        wav, sr = sf.read(bio, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {e}")

    # Convert to mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    # Convert to torch tensor
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)

    # Resample if needed
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        wav_tensor = resampler(wav_tensor)

    return wav_tensor  # shape (1, T)


def compute_embedding_from_tensor(wav_tensor: torch.Tensor):
    """
    Use SpeechBrain encoder to compute a speaker embedding.
    Returns a normalized numpy array (1D).
    """
    # EncoderClassifier accepts (batch, time) or (batch, channels, time) depending on model.
    # The spkrec-ecapa model expects a 1D waveform (batch, time).
    # Make sure tensor is on CPU (or GPU if you want to enable it).
    with torch.no_grad():
        # If tensor is shape (1, T), encoder.encode_batch expects (batch, samples)
        embeddings = encoder.encode_batch(wav_tensor)  # returns torch tensor (batch, embed_dim)
        emb = embeddings.squeeze(0).cpu().numpy()
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def save_audio_file(audio_dir: str, contents: bytes, original_name: str = None):
    fname = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(audio_dir, fname)
    with open(path, "wb") as f:
        f.write(contents)
    return path


def load_profile(user_dir: str):
    profile_path = os.path.join(user_dir, PROFILE_FILENAME)
    if not os.path.exists(profile_path):
        return None
    centroid = np.load(profile_path)
    meta = {}
    meta_path = os.path.join(user_dir, METADATA_FILENAME)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return {"centroid": centroid, "meta": meta}


def update_profile(user_dir: str, new_embedding: np.ndarray, saved_audio_relpaths: List[str] = None):
    """
    Append new embedding to embeddings.npy and recompute centroid.
    """
    emb_list_path = os.path.join(user_dir, EMB_LIST_FILENAME)
    if os.path.exists(emb_list_path):
        existing = np.load(emb_list_path)
        combined = np.vstack([existing, new_embedding.reshape(1, -1)])
    else:
        combined = new_embedding.reshape(1, -1)

    centroid = np.mean(combined, axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0:
        centroid = centroid / cnorm

    np.save(emb_list_path, combined)
    np.save(os.path.join(user_dir, PROFILE_FILENAME), centroid)

    # metadata
    meta = {
        "samples_total": int(combined.shape[0]),
        "audio_files": saved_audio_relpaths or []
    }
    with open(os.path.join(user_dir, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)

# make root redirect or serve index
@app.get("/")
def index():
    return FileResponse(os.path.join(ui_dir, "index.html"))

@app.post("/register/{username}")
async def register(username: str, files: List[UploadFile] = File(...)):
    """
    Register username with 1..N audio files (wav/mp3/...). Saves audio and embeddings locally.
    """
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 audio file required for enrollment.")

    user_dir, audio_dir = ensure_user_dirs(username)
    embeddings = []
    saved_relpaths = []

    for upload in files:
        data = await upload.read()
        saved_path = save_audio_file(audio_dir, data, original_name=upload.filename)
        saved_relpaths.append(os.path.relpath(saved_path, start=user_dir))

        wav_t = read_audio_bytes_to_tensor(data)        # torch tensor (1, T)
        emb = compute_embedding_from_tensor(wav_t)     # numpy 1D
        embeddings.append(emb)

    stacked = np.vstack(embeddings)
    centroid = np.mean(stacked, axis=0)
    if np.linalg.norm(centroid) > 0:
        centroid = centroid / np.linalg.norm(centroid)

    # Persist: combine with existing embeddings if present
    emb_list_path = os.path.join(user_dir, EMB_LIST_FILENAME)
    if os.path.exists(emb_list_path):
        existing = np.load(emb_list_path)
        combined = np.vstack([existing, stacked])
    else:
        combined = stacked

    np.save(emb_list_path, combined)
    np.save(os.path.join(user_dir, PROFILE_FILENAME), centroid)

    meta = {
        "username": username,
        "samples_total": int(combined.shape[0]),
        "audio_files": saved_relpaths
    }
    with open(os.path.join(user_dir, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)

    return JSONResponse({"ok": True, "username": username, "samples_stored": int(combined.shape[0])})


@app.post("/login/{username}")
async def login(username: str, file: UploadFile = File(...)):
    """
    Authenticate user with one audio file. Returns similarity and accepted boolean.
    """
    user_dir = os.path.join(DATA_DIR, username)
    profile = load_profile(user_dir)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile found for that username.")

    data = await file.read()
    wav_t = read_audio_bytes_to_tensor(data)
    emb = compute_embedding_from_tensor(wav_t)
    centroid = profile["centroid"]

    sim = float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid)))
    accepted = sim >= SIMILARITY_THRESHOLD

    attempts_log = os.path.join(user_dir, "attempts.log")
    with open(attempts_log, "a") as f:
        f.write(json.dumps({"similarity": sim, "accepted": accepted}) + "\n")

    return {"ok": accepted, "similarity": sim, "threshold": SIMILARITY_THRESHOLD}


@app.get("/profile/{username}")
async def get_profile(username: str):
    user_dir = os.path.join(DATA_DIR, username)
    profile = load_profile(user_dir)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile found.")
    meta_path = os.path.join(user_dir, METADATA_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return {"username": username, "meta": meta, "has_profile": True}
