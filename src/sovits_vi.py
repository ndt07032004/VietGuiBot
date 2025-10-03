# sovits_vi.py
import requests
import tempfile
import os
import wave
import numpy as np
import soundfile as sf
from src.logger import setup_logger

logger = setup_logger("SoVITS")

class TTSSoVITS:
    def __init__(self, base_url="http://127.0.0.1:9880"):
        self.url = base_url.rstrip("/")

    def __call__(self, text: str, voice: str = None, ref_audio: str = None):
        """
        Calls SoVITS http server, expects raw wav bytes in response.content or full wav file.
        Returns: {"audio": np.ndarray, "sampling_rate": int}
        """
        data = {"text": text, "text_language": "vi", "cut_punc": ",."}
        try:
            resp = requests.post(self.url, json=data, timeout=20)
            if resp.status_code != 200:
                logger.error(f"SoVITS error code: {resp.status_code} {resp.text}")
                return {"audio": np.zeros(1, dtype=np.float32), "sampling_rate": 16000}

            # Save to temp file then read
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(resp.content)
                tmp_path = tf.name
            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Normalize
            maxv = float(np.max(np.abs(audio))) if audio.size else 0.0
            if maxv > 0:
                audio = audio / maxv

            return {"audio": audio.astype(np.float32), "sampling_rate": int(sr)}
        except Exception as e:
            logger.error(f"SoVITS call failed: {e}")
            return {"audio": np.zeros(1, dtype=np.float32), "sampling_rate": 16000}
