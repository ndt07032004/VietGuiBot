from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from src.llm_rag import generate_response
from src.asr import transcribe_audio, record_audio
from src.tts import synthesize_speech
from src.logger import logger
import io
import asyncio

app = FastAPI()

# Thêm CORS cho cross-origin (mượt với Unity)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def create_app(config, rag_chain, asr_model, tts_model):
    API_KEY = config['human_integration']['api_key']
    
    @app.get("/")
    async def index():
        """Trả về giao diện chat.html"""
        with open('static/chat.html', 'r') as f:
            return HTMLResponse(f.read())
    
    @app.post("/v1/chat/completions")  # Tương thích OpenAI cho Unity Human
    async def chat_completions(request: Request):
        """API chat với streaming SSE.
        Cải tiến: Auth, async stream response để mượt (không chờ full).
        """
        auth = request.headers.get('Authorization')
        if auth != f"Bearer {API_KEY}":
            return {"error": "Unauthorized"}, 401
        
        data = await request.json()
        messages = data.get('messages', [])
        input_text = messages[-1]['content'] if messages else ""
        
        # Streaming response
        async def generate():
            async for chunk in generate_response(rag_chain, input_text):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    @app.post("/asr")
    async def asr_endpoint():
        """Endpoint ASR real-time.
        Cải tiến: Async để không block server.
        """
        audio_data = await asyncio.to_thread(record_audio)  # Offload to thread
        text = transcribe_audio(asr_model, audio_data)
        return {"text": text}
    
    @app.post("/tts")
    async def tts_endpoint(request: Request):
        """Endpoint TTS với streaming audio.
        Cải tiến: Stream audio chunks cho play real-time.
        """
        data = await request.json()
        text = data['text']
        output_path = f"./audio/response_{hash(text)}.wav"
        await synthesize_speech(asr_model, text, output_path)  # Async
        with open(output_path, 'rb') as f:
            return StreamingResponse(io.BytesIO(f.read()), media_type='audio/wav')  # Stream nếu cần chunk
    
    # WebSocket cho Unity Human real-time (mượt hơn HTTP)
    @app.websocket("/ws/human")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket cho streaming real-time với Unity.
        Cải tiến: Gửi/receive audio/text chunks liên tục.
        """
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            input_text = data.get('message')
            async for chunk in generate_response(rag_chain, input_text):
                await websocket.send_json({"chunk": chunk})
    
    return app