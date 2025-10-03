import numpy as np
import soundfile as sf
from src.logger import logger
from transformers import pipeline as hf_pipeline

# Import các backend TTS
from src.tts_f5 import TTSF5
from src.sovits_vi import TTSSoVITS
from src.sovits_vi_v3 import TTSSoVITSv3


async def synthesize_speech(model, text: str, output_path: str) -> str:
    """
    Gọi model TTS (F5, SoVITS, HuggingFace) → chuẩn hóa → ghi file WAV.
    Trả về đường dẫn file WAV.
    """
    try:
        logger.debug(f"🔊 Synthesizing speech for text: {text}")

        # HuggingFace pipeline trả về list
        if callable(model) and not hasattr(model, "synthesize"):
            result = model(text)
            if isinstance(result, list):
                result = result[0]
        else:
            result = model(text)

        audio = result["audio"]
        sampling_rate = result["sampling_rate"]

        # chuẩn hóa dữ liệu âm thanh
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        sf.write(output_path, audio.astype(np.float32), sampling_rate)
        logger.info(f"✅ TTS generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


async def init_tts(config):
    """
    Khởi tạo backend TTS dựa vào config.
    """
    model_name = config["tts"]["model"]

    if model_name == "f5-tts":
        logger.info("🔧 Init F5-TTS backend")
        return TTSF5()
    elif model_name == "sovits":
        logger.info("🔧 Init SoVITS backend")
        return TTSSoVITS()
    elif model_name == "sovitsv3":
        logger.info("🔧 Init SoVITSv3 backend")
        return TTSSoVITSv3()
    else:
        # HuggingFace pipeline
        if model_name.strip().lower() == "huggingface":
            logger.warning("⚠️ Không có model cụ thể, fallback sang 'facebook/mms-tts'")
            model_name = "facebook/mms-tts"

        logger.info(f"🔧 Load HuggingFace TTS model: {model_name}")
        pipe = hf_pipeline("text-to-speech", model=model_name)

        # Bọc lại để luôn trả về audio dict
        def hf_wrapper(text: str):
            result = pipe(text, generate_kwargs={"lang": "vie"})
            if isinstance(result, list):
                result = result[0]
            return {
                "audio": result["audio"],
                "sampling_rate": result["sampling_rate"]
            }

        return hf_wrapper
