import os
import io
import json
import asyncio
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.llm_rag import generate_response
from src.asr import transcribe_audio, record_audio
from src.tts import synthesize_speech
from src.logger import logger


def create_app(config, rag_chain, asr_model, tts_model):
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    API_KEY = config['human_integration']['api_key']

    @app.get("/")
    async def index():
        """Trả về giao diện chat.html"""
        # Lấy đường dẫn tuyệt đối tới thư mục static
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent folder của src
        path = os.path.join(base_dir, "static", "chat.html")

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File chat.html không tồn tại")

        with open(path, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """API chat (OpenAI compatible) với streaming SSE"""
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")

        data = await request.json()
        messages = data.get("messages", [])
        input_text = messages[-1]['content'] if messages else ""

        async def generate():
            async for chunk in generate_response(rag_chain, input_text):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/asr")
    async def asr_endpoint():
        """ASR real-time"""
        audio_data = await asyncio.to_thread(record_audio)
        text = transcribe_audio(asr_model, audio_data)
        return {"text": text}

    @app.post("/tts")
    async def tts_endpoint(request: Request):
        """TTS real-time"""
        data = await request.json()
        text = data['text']
        output_path = f"./audio/response_{hash(text)}.wav"

        await synthesize_speech(tts_model, text, output_path)

        with open(output_path, "rb") as f:
            return StreamingResponse(io.BytesIO(f.read()), media_type="audio/wav")

    @app.websocket("/ws/human")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket cho Unity real-time"""
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            input_text = data.get("message")
            async for chunk in generate_response(rag_chain, input_text):
                await websocket.send_json({"chunk": chunk})

    return app
