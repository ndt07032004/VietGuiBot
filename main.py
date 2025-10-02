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
config_path = "system.conf"

# Load config với error handling
try:
    with open(config_path, 'r', encoding='utf-8') as f:
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
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])