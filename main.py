import json
import os
import asyncio
import subprocess
import time
import requests
from dotenv import load_dotenv
from src.logger import setup_logger
from src.api import create_app
from src.llm_rag import init_rag
from src.asr import init_asr
from src.tts import init_tts
import uvicorn

load_dotenv()
logger = setup_logger("VietGuiBot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "system.conf")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file config: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

logger.info("🚀 Khởi động VietGuiBot...")

def start_sovits_server():
    try:
        r = requests.get("http://127.0.0.1:9880", timeout=2)
        if r.status_code == 200:
            logger.info("✅ SoVITS server đã chạy sẵn.")
            return None
    except Exception:
        pass

    logger.info("🟡 Chưa thấy SoVITS server, khởi động...")
    proc = subprocess.Popen(
        ["python", "sovits_server.py", "--port", "9880"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)
    logger.info("✅ SoVITS server đã khởi động.")
    return proc

async def async_init():
    qa_chain = init_rag(config)
    asr_model = await init_asr(config)

    if config["tts"]["model"] in ["sovits", "sovitsv3"]:
        start_sovits_server()

    tts_model = await init_tts(config)
    return qa_chain, asr_model, tts_model

loop = asyncio.get_event_loop()
qa_chain, asr_model, tts_model = loop.run_until_complete(async_init())

app = create_app(config, qa_chain, asr_model, tts_model)

if __name__ == "__main__":
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
