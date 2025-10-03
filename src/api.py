from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.llm_rag import generate_response
from src.asr import transcribe_audio, record_audio
from src.tts import synthesize_speech
from src.logger import logger
import io
import asyncio
import os
import json
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def create_app(config, qa_chain, asr_model, tts_model):
    @app.get("/")
    async def index():
        with open('static/chat.html', 'r', encoding="utf-8") as f:
            return HTMLResponse(f.read())

    # ✅ Sửa lại: Thêm kiểm tra file tồn tại
    @app.get("/get")
    async def get_response(request: Request):
        msg = request.query_params.get("msg", "")
        if not msg:
            return JSONResponse({"error": "No message provided"})
        
        # Tạo response text từ LLM
        response_text = generate_response(qa_chain, msg)
        logger.info(f"[GET] User: {msg}")
        logger.info(f"[GET] Bot text: {response_text}")

        # Tạo audio từ response text
        output_path = f"./audio/response_{hash(response_text)}.wav"
        audio_path = await synthesize_speech(tts_model, response_text, output_path)
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file not created: {audio_path}")
            return JSONResponse({"answer": response_text, "error": "Failed to generate audio"})

        # Đọc file audio và encode base64
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        logger.info(f"[GET] Bot audio generated: {audio_path}")
        return JSONResponse({
            "answer": response_text,
            "audio": audio_base64
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        data = await request.json()
        messages = data.get('messages', [])
        input_text = messages[-1]['content'] if messages else ""
        response = generate_response(qa_chain, input_text)
        logger.info(f"[POST] User: {input_text}")
        logger.info(f"[POST] Bot: {response}")
        return {"choices": [{"message": {"content": response}}]}

    @app.post("/asr")
    async def asr_endpoint():
        audio_data = await asyncio.to_thread(record_audio)
        text = transcribe_audio(asr_model, audio_data)
        return {"text": text}

    @app.post("/tts")
    async def tts_endpoint(request: Request):
        data = await request.json()
        text = data['text']
        output_path = f"./audio/response_{hash(text)}.wav"
        await synthesize_speech(tts_model, text, output_path)
        with open(output_path, 'rb') as f:
            return StreamingResponse(io.BytesIO(f.read()), media_type="audio/wav")

    @app.websocket("/ws/human")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            input_text = data.get('message')
            response = generate_response(qa_chain, input_text)
            await websocket.send_json({"chunk": response})

    return app