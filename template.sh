#!/bin/bash
set -e

echo "🚀 Đang tạo cấu trúc project Vietnamese Digital Bot..."

# Tạo thư mục chính
mkdir -p src
mkdir -p static
mkdir -p data

# Tạo file ở root
touch main.py
touch system.conf
touch requirements.txt
touch store_index.py
touch .env

# Tạo file trong src
touch src/__init__.py
touch src/asr.py
touch src/tts.py
touch src/llm_rag.py
touch src/api.py
touch src/helper.py
touch src/logger.py
touch src/prompt.py

# Tạo file trong static
touch static/style.css
touch templates/chat.html

echo "✅ Project structure created successfully."
