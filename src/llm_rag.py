# src/llm_rag.py

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from src.prompt import system_prompt

import os

load_dotenv()


def init_rag(config):
    # --- API Key từ config ---
    pinecone_api_key = config["pinecone"]["api_key"]
    if not pinecone_api_key:
        raise ValueError("❌ PINECONE_API_KEY not found in config")

    # --- Pinecone client ---
    pc = PineconeClient(api_key=pinecone_api_key)

    # --- Lấy tên index ---
    index_name = config["pinecone"]["index_name"]
    spec = ServerlessSpec(cloud="gcp", region="us-west1")

    # --- Tạo index nếu chưa có ---
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 embeddings
            metric="cosine",
            spec=spec
        )

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # --- Lấy instance index ---
    index = pc.Index(index_name)

    # --- Vectorstore & retriever ---
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- LLM (Ollama) ---
    llm = ChatOllama(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"]
    )

    # --- Prompt tuỳ chỉnh ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # --- Chain xử lý document ---
    combine_chain = create_stuff_documents_chain(llm, prompt)

    # --- Chain cuối cùng (Retrieval + QA) ---
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    return rag_chain


def generate_response(rag_chain, input_text: str):
    # invoke thay vì run → tránh lỗi input_documents
    response = rag_chain.invoke({"input": input_text})
    return response.get("answer", response)
