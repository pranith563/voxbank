# audio_utils.py
import numpy as np
import webrtcvad
import torch
import io
import soundfile as sf
import torchaudio

def read_wav_bytes_to_tensor(wav_bytes: bytes, target_sr=16000):
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, T)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor = resampler(tensor)
    return tensor.squeeze(0).numpy(), target_sr  # 1D np array, sr

def vad_trim(waveform: np.ndarray, sr: int, mode:int=3):
    # waveform: float32 np mono in [-1,1]
    vad = webrtcvad.Vad(mode)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    # convert to 16-bit PCM for VAD
    pcm16 = (waveform * 32767).astype(np.int16).tobytes()
    frames = []
    for i in range(0, len(pcm16), frame_len*2):
        chunk = pcm16[i:i+frame_len*2]
        if len(chunk) < frame_len*2:
            break
        is_speech = vad.is_speech(chunk, sample_rate=sr)
        frames.append((chunk, is_speech))
    # reconstruct speech frames
    speech_samples = []
    for chunk, is_speech in frames:
        if is_speech:
            # convert back to float samples
            speech_samples.append(np.frombuffer(chunk, dtype=np.int16).astype(np.float32)/32767.0)
    if len(speech_samples)==0:
        return waveform  # fallback: return original if VAD removed everything
    return np.concatenate(speech_samples)

def rms_normalize(waveform: np.ndarray, target_rms=0.1, eps=1e-6):
    rms = np.sqrt(np.mean(waveform**2))
    if rms < eps:
        return waveform
    return waveform * (target_rms / (rms + 1e-9))
