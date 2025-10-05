import asyncio
import os
from src.tts import init_tts, synthesize_speech

# Config giống file JSON bạn có
config = {
    "tts": {
        "model": "facebook/mms-tts-vie",   # model tiếng Việt
        "output_dir": "./audio"            # thư mục lưu file wav
    }
}

async def main():
    # Khởi tạo TTS pipeline
    tts_model = await init_tts(config)

    # Text để chuyển thành giọng nói
    text = "Xin chào các bạn, mình là bot hướng dẫn viên du lịch."

    # Đường dẫn file đầu ra
    output_dir = config["tts"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo.wav")

    # Sinh audio
    file_path = await synthesize_speech(tts_model, text, output_path)

    if file_path:
        print("✅ Đã tạo file:", file_path)
    else:
        print("❌ Tạo file thất bại")

if __name__ == "__main__":
    asyncio.run(main())
