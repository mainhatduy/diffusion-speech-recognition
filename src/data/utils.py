import io
import re
import wave
import unicodedata
import numpy as np

def _decode_wav_bytes(wav_bytes: bytes):
    """Decode raw WAV bytes to float32 numpy array using python's built-in wave module.
    This avoids dependency on torchcodec/soundfile/librosa."""
    f = wave.open(io.BytesIO(wav_bytes), 'rb')
    n_channels = f.getnchannels()
    sampwidth = f.getsampwidth()
    n_frames = f.getnframes()
    sample_rate = f.getframerate()
    raw_frames = f.readframes(n_frames)
    f.close()
    
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    data = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)
    
    # Normalize to [-1.0, 1.0]
    if sampwidth == 2:
        data = data / 32768.0
    elif sampwidth == 4:
        data = data / 2147483648.0
    
    # Convert stereo to mono by averaging channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    
    return data, sample_rate

def normalize_text(text: str) -> str:
    """Normalize and clean text by lowercasing and removing punctuation/symbols."""
    if not text:
        return ""
    normalized_text = text.lower()
    chars = []
    for char in normalized_text:
        cat = unicodedata.category(char)
        if cat.startswith('P') or cat.startswith('S'):
            chars.append(' ')
        else:
            chars.append(char)
    normalized_text = "".join(chars)
    return re.sub(r"\s+", " ", normalized_text).strip()
