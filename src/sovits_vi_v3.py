# sovits_vi_v3.py
import requests
import tempfile
import os
import numpy as np
import soundfile as sf
from src.logger import setup_logger

logger = setup_logger("SoVITSv3")

class TTSSoVITSv3:
    def __init__(self, base_url="http://127.0.0.1:9880/tts"):
        self.url = base_url

    def __call__(self, text: str, voice: str = None, ref_audio: str = None):
        data = {
            "text": text,
            "text_lang": "vi",
            "ref_audio_path": ref_audio or "./samples/ref.wav",
            "prompt_text": "Xin chào, tôi là hướng dẫn viên",
            "prompt_lang": "vi",
            "top_k": 5, "top_p": 1, "temperature": 1, "text_split_method": "cut5",
            "batch_size": 1, "batch_threshold": 0.75, "split_bucket": True,
            "speed_factor": 1.0, "fragment_interval": 0.3, "seed": -1,
            "media_type": "wav", "streaming_mode": False, "parallel_infer": True,
            "repetition_penalty": 1.2
        }
        try:
            resp = requests.post(self.url, json=data, timeout=30)
            if resp.status_code != 200:
                logger.error(f"SoVITSv3 error: {resp.status_code} {resp.text}")
                return {"audio": np.zeros(1, dtype=np.float32), "sampling_rate": 32000}

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(resp.content)
                tmp_path = tf.name

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            maxv = float(np.max(np.abs(audio))) if audio.size else 0.0
            if maxv > 0:
                audio = audio / maxv

            return {"audio": audio.astype(np.float32), "sampling_rate": int(sr)}
        except Exception as e:
            logger.error(f"SoVITSv3 call failed: {e}")
            return {"audio": np.zeros(1, dtype=np.float32), "sampling_rate": 32000}
