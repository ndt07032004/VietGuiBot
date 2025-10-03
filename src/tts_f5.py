# tts_f5.py
import soundfile as sf
from src.logger import setup_logger
from src.f5_tts_vi import F5Engine

import numpy as np

logger = setup_logger("TTSF5")

class TTSF5:
    def __init__(self, **kwargs):
        self.engine = F5Engine()

    def __call__(self, text: str, voice: str = None, ref_audio: str = None):
        """
        Return: {"audio": np.ndarray(float32), "sampling_rate": int}
        """
        audio, sr = self.engine.synthesize_to_array(text, ref_audio_path=ref_audio)
        if audio is None:
            logger.error("F5 returned no audio.")
            return {"audio": np.zeros(1, dtype=np.float32), "sampling_rate": 22050}
        return {"audio": audio.astype(np.float32), "sampling_rate": int(sr)}
