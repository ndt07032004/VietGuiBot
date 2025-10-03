import json
import os
import asyncio  # Cho async init
from dotenv import load_dotenv
from src.logger import setup_logger
from src.api import create_app
from src.llm_rag import init_rag
from src.asr import init_asr
from src.tts import init_tts
import uvicorn  # Server cho FastAPI

load_dotenv()

logger = setup_logger("VietnameseDigitalBot")
config_path = "system.json"

# Config mặc định (nếu chưa có file)
default_config = {
    "server": {"host": "0.0.0.0", "port": 8080, "debug": True},
    "llm": {
        "model": "qwen2:7b",
        "base_url": "http://localhost:11434",
        "system_prompt": "Bạn là TourGuideBot, hướng dẫn viên du lịch Việt Nam.",
        "streaming": True,
    },
    "asr": {"model": "small", "language": "vi", "device": "cpu"},
    "tts": {
        "model": "facebook/mms-tts-vie",
        "output_dir": "./audio/",
        "streaming": True,
    },
    "pinecone": {
        "index_name": "digital-hunman",
        "api_key": "pcsk_6BZpRA_QgbTMLShrvgfY8PC1w3fzYkDaQB39144GvKCCSNZgGDjM5JjjYDtmviNVxH8KdZ"
    },
    "human_integration": {"api_key": "simple_auth_key_123", "streaming": True, "websocket": True},
    "interfaces": {"text": True, "voice": True, "digital_hunman": True},
}

# Load config với error handling
if not os.path.exists(config_path):
    logger.warning(f"{config_path} không tồn tại, tạo file mặc định...")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    config = default_config
else:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Config loaded successfully")
    except Exception as e:
        logger.error(f"Config load error: {e}")
        raise

logger.info("Khởi động Vietnamese Digital Bot với cải tiến real-time...")

# Async init modules để không block (mượt hơn khi load model lớn)
async def async_init():
    rag_chain = init_rag(config)
    asr_model = await init_asr(config)  # Async load
    tts_model = await init_tts(config)  # Async load
    return rag_chain, asr_model, tts_model

# Chạy async init
loop = asyncio.get_event_loop()
rag_chain, asr_model, tts_model = loop.run_until_complete(async_init())

# Tạo và chạy FastAPI app
app = create_app(config, rag_chain, asr_model, tts_model)

# Chạy server với uvicorn (hỗ trợ async, mượt hơn Flask)
if __name__ == "__main__":
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
