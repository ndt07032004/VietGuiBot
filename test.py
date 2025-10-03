import asyncio
from src.tts import init_tts, synthesize_speech

config = {"tts": {"model": "facebook/mms-tts-vie", "output_dir": "./audio/"}}

async def test():
    tts_model = await init_tts(config)
    output_path = await synthesize_speech(tts_model, "Xin chào, tôi là VietGuiBot!", "./audio/test.wav")
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    asyncio.run(test())