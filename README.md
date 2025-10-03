# VietGuiBot
# VietGuiBot
VietGuiBot là chatbot hướng dẫn viên du lịch Việt Nam sử dụng RAG với Pinecone và Ollama Qwen2:7b.

## Tính năng
- ASR tiếng Việt (RealtimeSTT/Whisper).
- TTS tiếng Việt (viXTTS/f5-tts).
- Giao diện web (FastAPI + HTML).
- Tích hợp Unity Human cho digital avatar.

## Cài đặt
1. Clone repo: `git clone https://github.com/ndt07032004/VietGuiBot.git`
2. Tạo environment: `conda create -n historybot python=3.10`
3. Cài dependencies: `pip install -r requirements.txt`
4. Chạy: `python main.py`
5. Truy cập: http://localhost:5000

## Requirements
- Ollama với model qwen2:7b.
- Pinecone API key trong .env.