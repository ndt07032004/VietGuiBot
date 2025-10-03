# api.py
import io
import os
import base64
import asyncio
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.logger import setup_logger
import soundfile as sf
import numpy as np

logger = setup_logger("API")

def create_app(config, qa_chain, asr_model, tts_model):
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.get("/")
    async def index():
        if os.path.exists('static/chat.html'):
            with open('static/chat.html', 'r', encoding="utf-8") as f:
                return HTMLResponse(f.read())
        return JSONResponse({"error": "Chat interface not found"})


    @app.get("/get")
    async def get_response(request: Request):
        msg = request.query_params.get("msg", "")
        if not msg:
            return JSONResponse({"error": "No message provided"})
        try:
            # gọi LLM / RAG
            response_text = qa_chain(msg) if callable(qa_chain) else str(qa_chain)
            logger.info(f"[GET] User: {msg}")
            logger.info(f"[GET] Bot text: {response_text}")

            # synthesize audio
            output_dir = config.get("tts", {}).get("output_dir", "./audio/")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"response_{abs(hash(response_text))}.wav")

            if asyncio.iscoroutinefunction(tts_model):
                tts_result = await tts_model(response_text)
            else:
                loop = asyncio.get_event_loop()
                tts_result = await loop.run_in_executor(None, lambda: tts_model(response_text))

            audio = tts_result.get("audio")
            sr = int(tts_result.get("sampling_rate", 22050))
            if audio is None:
                return JSONResponse({"answer": response_text, "error": "TTS failed"})

            sf.write(output_path, audio.astype("float32"), sr)
            with open(output_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            return JSONResponse({"answer": response_text, "audio": audio_base64})
        except Exception as e:
            logger.exception(f"/get failed: {e}")
            return JSONResponse({"error": str(e)})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        data = await request.json()
        messages = data.get('messages', [])
        input_text = messages[-1]['content'] if messages else ""
        try:
            response = qa_chain(input_text) if callable(qa_chain) else str(qa_chain)
            logger.info(f"[POST] User: {input_text}")
            logger.info(f"[POST] Bot: {response}")
            return {"choices": [{"message": {"content": response}}]}
        except Exception as e:
            logger.exception("chat_completions error")
            return {"error": str(e)}

    @app.post("/asr")
    async def asr_endpoint():
        # Keep your original record_audio/transcribe_audio implementation
        try:
            if hasattr(asr_model, "record_audio"):
                audio_data = await asyncio.to_thread(asr_model.record_audio)
            else:
                # If you provided a function record_audio elsewhere
                audio_data = None
            if hasattr(asr_model, "transcribe"):
                text = asr_model.transcribe(audio_data)
            else:
                text = "ASR not implemented"
            return {"text": text}
        except Exception as e:
            logger.exception("ASR endpoint error")
            return {"text": ""}

    @app.post("/tts")
    async def tts_endpoint(request: Request):
        data = await request.json()
        text = data.get('text', '')
        voice = data.get('voice', None)
        ref_audio = data.get('ref_audio', None)
        if not text:
            return JSONResponse({"error": "No text provided"})
        try:
            output_dir = config.get("tts", {}).get("output_dir", "./audio/")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"response_{abs(hash(text))}.wav")

            if asyncio.iscoroutinefunction(tts_model):
                tts_result = await tts_model(text, voice=voice, ref_audio=ref_audio)
            else:
                loop = asyncio.get_event_loop()
                tts_result = await loop.run_in_executor(None, lambda: tts_model(text, voice=voice, ref_audio=ref_audio))

            audio = tts_result.get("audio")
            sr = int(tts_result.get("sampling_rate", 22050))
            if audio is None:
                return JSONResponse({"error": "TTS failed"})

            sf.write(output_path, audio.astype("float32"), sr)
            with open(output_path, 'rb') as f:
                return StreamingResponse(io.BytesIO(f.read()), media_type="audio/wav")
        except Exception as e:
            logger.exception("TTS endpoint error")
            return JSONResponse({"error": str(e)})

    @app.websocket("/ws/human")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                input_text = data.get('message')
                response = qa_chain(input_text) if callable(qa_chain) else str(qa_chain)
                await websocket.send_json({"chunk": response})
        except Exception as e:
            logger.info(f"Websocket closed: {e}")

    return app
