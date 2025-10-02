from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.logger import logger
import redis.asyncio as redis
import json
import os
import asyncio
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

async def init_rag(config: dict):
    """
    Khởi tạo RAG chain với Pinecone và Parent Document Retriever.
    Cải tiến:
    - Sử dụng Pinecone thay Chroma, đảm bảo index tồn tại.
    - Thêm caching với Redis để giảm latency.
    - Áp dụng Parent Document Retriever để tránh cắt nội dung.
    """
    # Load embeddings
    embeddings = download_hugging_face_embeddings()
    
    # Khởi tạo Pinecone client
    pinecone_api_key = os.getenv("PINECONE_API_KEY") or config['pinecone']['api_key']
    if not pinecone_api_key:
        logger.error("Pinecone API key not found in .env or config")
        raise ValueError("Pinecone API key is required")
    
    # Kiểm tra và tạo index nếu chưa tồn tại
    index_name = config['pinecone']['index_name']
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Từ model embedding
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Vector store cho child chunks
    docsearch = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": config.get('rag', {}).get('retriever_k', 3)})
    
    # Khởi tạo Parent Document Retriever
    child_vectorstore = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)
    parent_docstore = InMemoryStore()  # Lưu parent chunks
    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        search_kwargs={"k": config.get('rag', {}).get('retriever_k', 5)}  # Tăng k để lấy nhiều chunk
    )
    
    # LLM với streaming
    chat_model = ChatOllama(
        model=config['llm']['model'],
        base_url=config['llm']['base_url'],
        stream=config['llm']['streaming']
    )
    
    # Prompt yêu cầu trả lời đầy đủ
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nHãy trả lời đầy đủ và chi tiết dựa trên toàn bộ ngữ cảnh sau: {context}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Khởi tạo Redis cache
    cache = None
    if config.get('cache', {}).get('enabled', False):
        cache = redis.from_url(config['cache']['redis_url'])
        logger.info("Redis cache initialized")
    
    logger.info(f"RAG chain initialized with Pinecone and streaming")
    return rag_chain, cache

async def generate_response(rag_chain, cache, input_text: str):
    """
    Tạo phản hồi từ RAG asynchronously với streaming.
    Cải tiến:
    - Kiểm tra cache để giảm latency.
    - Yield chunks để stream text (mượt cho UI).
    - Đảm bảo input từ ASR được xử lý đầy đủ.
    """
    try:
        if not input_text:
            logger.warning("Empty input from ASR")
            yield "Vui lòng nói lại rõ ràng hơn."
            return
        
        # Kiểm tra cache
        if cache:
            cached_response = await cache.get(f"rag:{input_text}")
            if cached_response:
                logger.info("Cache hit")
                for chunk in json.loads(cached_response).split(" "):
                    yield chunk + " "
                return
        
        # Generate response với streaming
        response = ""
        async for chunk in rag_chain.astream({"input": input_text}):
            response += chunk.get("answer", "")
            yield chunk.get("answer", "")
        
        # Lưu cache (TTL 1 giờ)
        if cache:
            await cache.set(f"rag:{input_text}", json.dumps(response), ex=3600)
            
    except Exception as e:
        logger.error(f"RAG error: {e}")
        yield "Xin lỗi, tôi chưa biết thông tin này."