from transformers import pipeline
import torch
import os
import asyncio
import soundfile as sf
from src.logger import logger


async def init_tts(config: dict):
    """
    Khởi tạo TTS pipeline từ HuggingFace.
    Ưu tiên: facebook/mms-tts-vie (tiếng Việt) → fallback sang Bark.
    """
    model_name = config["tts"].get("model", "facebook/mms-tts-vie")
    output_dir = config["tts"].get("output_dir", "outputs/tts")
    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.debug(f"🔊 Loading TTS model: {model_name}")
        tts = pipeline(
            "text-to-speech",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info(f"✅ Loaded TTS model: {model_name}")
        return tts
    except Exception as e:
        logger.error(f"❌ Failed to load TTS model {model_name}: {e}")
        # fallback sang Bark
        if model_name != "suno/bark-small":
            logger.info("👉 Falling back to Bark (suno/bark-small)")
            config["tts"]["model"] = "suno/bark-small"
            return await init_tts(config)
        raise


async def synthesize_speech(model, text: str, output_path: str) -> str:
    """
    Tạo audio từ text tiếng Việt.
    Lưu file .wav để phát ngay sau khi generate.
    """
    try:
        logger.debug(f"🔊 Synthesizing speech for text: {text}")
        result = model(text)

        # HuggingFace pipeline trả về mảng numpy chứa audio
        audio = result["audio"]
        sampling_rate = result["sampling_rate"]

        # lưu file wav
        sf.write(output_path, audio, sampling_rate)
        logger.info(f"✅ TTS generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
