import json
import os
import asyncio
from dotenv import load_dotenv
from src.logger import setup_logger
from src.api import create_app
from src.llm_rag import init_rag
from src.asr import init_asr
from src.tts import init_tts
import uvicorn

load_dotenv()

logger = setup_logger("VietGuiBot")


config_path = 'system.conf'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)


logger.info("Khởi động VietGuiBot...")

async def async_init():
    qa_chain = init_rag(config)
    asr_model = await init_asr(config)
    tts_model = await init_tts(config)
    return qa_chain, asr_model, tts_model

loop = asyncio.get_event_loop()
qa_chain, asr_model, tts_model = loop.run_until_complete(async_init())

app = create_app(config, qa_chain, asr_model, tts_model)

if __name__ == "__main__":
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])