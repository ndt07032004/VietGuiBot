# src/asr.py
import pyaudio
import numpy as np
from typing import Optional
import asyncio
from src.logger import setup_logger
import whisper
import webrtcvad
import torch

logger = setup_logger("ASR")

async def init_asr(config: dict):
    """
    Khởi tạo model ASR real-time bằng Whisper + VAD.
    """
    try:
        model_name = config.get('asr', {}).get('model', 'small')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper model {model_name} loaded on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper ASR: {e}")
        raise

async def transcribe_audio(model, sample_rate: int = 16000) -> Optional[str]:
    """
    Chuyển đổi audio thành text tiếng Việt real-time.
    """
    try:
        audio_data = await record_audio(sample_rate)
        if audio_data is None:
            return None

        # Whisper expects either path or numpy float32 array; here we use numpy
        result = model.transcribe(audio_data, language="vi")
        text = result.get("text", "").strip()
        logger.info(f"Transcription: {text}")
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None

async def record_audio(sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Ghi âm từ microphone với VAD (webrtcvad).
    """
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=512
        )
        frames = []
        vad = webrtcvad.Vad(mode=3)  # 0-3
        logger.info("🎙️ Bắt đầu ghi âm... (tự dừng khi im lặng)")

        while True:
            data = await asyncio.to_thread(stream.read, 512, False)
            is_speech = vad.is_speech(data, sample_rate)
            if is_speech:
                frames.append(np.frombuffer(data, dtype=np.int16))
            else:
                if len(frames) > 0:
                    break
            # Safety max 15s
            if len(frames) * 512 / sample_rate > 15:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if not frames:
            return None

        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        return audio
    except Exception as e:
        logger.error(f"Recording error: {e}")
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()
        return None
