import numpy as np
import soundfile as sf
from src.logger import logger

async def synthesize_speech(model, text: str, output_path: str) -> str:
    """
    Tạo audio từ text tiếng Việt bằng HuggingFace TTS.
    Fix lỗi ushort bằng chuẩn hóa float32 [-1,1].
    """
    try:
        logger.debug(f"🔊 Synthesizing speech for text: {text}")
        result = model(text)

        # HF pipeline trả về numpy array
        audio = result["audio"]
        sampling_rate = result["sampling_rate"]

        # Ép float32, loại NaN/Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val  # normalize [-1,1]

        # Ghi file wav an toàn
        sf.write(output_path, audio.astype(np.float32), sampling_rate)
        logger.info(f"✅ TTS generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
