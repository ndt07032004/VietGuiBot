# f5_tts_vi.py
import os
import time
import tempfile
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from src.logger import setup_logger

logger = setup_logger("F5TTS")

# NOTE: adjust imports for the F5 TTS repo you use.
# The original code used `from f5_tts.infer import utils_infer`.
# Keep the same if your environment has it.
try:
    from f5_tts.infer import utils_infer
except Exception:
    utils_infer = None
    logger.warning("f5_tts.infer.utils_infer not available. Ensure F5-TTS repo is installed.")

class F5Engine:
    def __init__(self):
        if utils_infer is None:
            raise RuntimeError("F5-TTS inference utils not found.")
        # Example model init — adapt args if your F5 repo differs
        logger.info("Loading F5 TTS model...")
        self.ckpt_path = hf_hub_download("zalopay/vietnamese-tts", "model_960000.pt")
        self.vocab_file = hf_hub_download("zalopay/vietnamese-tts", "vocab.txt")

# Truyền ckpt_path khi load model
        self.model = utils_infer.load_model(
            "DiT",
            dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=self.ckpt_path
            )
        self.vocoder = utils_infer.load_vocoder()

        # optional: checkpoint/vocab if needed by your version
        try:
            self.ckpt_path = hf_hub_download("zalopay/vietnamese-tts", "model_960000.pt")
            self.vocab_file = hf_hub_download("zalopay/vietnamese-tts", "vocab.txt")
        except Exception as e:
            logger.debug(f"HF download skipped or failed: {e}")

    def synthesize_to_array(self, text: str, ref_audio_path: str = None):
        """
        Returns: (audio_np: np.ndarray(float32), sampling_rate: int)
        """
        try:
            # If your utils_infer provides a direct numpy output, use it. Otherwise produce bytes then read.
            # Using preprocess_ref_audio_text if available (original code used it)
            if ref_audio_path and hasattr(utils_infer, "preprocess_ref_audio_text"):
                ref_audio, ref_text = utils_infer.preprocess_ref_audio_text(ref_audio_path, "xin chào, tôi là hướng dẫn viên")
            else:
                ref_audio, ref_text = None, None

            final_wave, sr, _ = utils_infer.infer_process(
                ref_audio, ref_text, text, self.model, self.vocoder,
                cross_fade_duration=0.15, nfe_step=32, speed=1.0
            )
            # final_wave may be bytes or numpy depending on utils_infer implementation.
            # Normalize to numpy float32 array and determine sr.
            if isinstance(final_wave, (bytes, bytearray)):
                # write to temp file and read with soundfile to get array + sr
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(final_wave)
                    tmp_path = tf.name
                audio, sampling_rate = sf.read(tmp_path, dtype="float32")
                os.unlink(tmp_path)
            else:
                # assume numpy array
                audio = np.array(final_wave, dtype=np.float32)
                # If sr returned from infer_process use it, else default 22050
                sampling_rate = sr or 22050

            # Mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Normalize
            max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
            if max_val > 0:
                audio = audio / max_val

            return audio.astype(np.float32), int(sampling_rate)
        except Exception as e:
            logger.error(f"F5 synth error: {e}")
            return None, None

if __name__ == "__main__":
    eng = F5Engine()
    a, sr = eng.synthesize_to_array("Xin chào, tôi là GuideBot.")
    print("len", None if a is None else a.shape, "sr", sr)
