import os
import shutil
from datetime import datetime
from typing import List
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient

import fitz  # PyMuPDF (REMPLACEMENT)

import re
import unicodedata
from dotenv import load_dotenv

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel

# ---------------- LOAD ENV ----------------
load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------- EMBEDDINGS ----------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    connection_args={
        "uri": ZILLIZ_URI,
        "token": ZILLIZ_TOKEN
    },
    auto_id=True
)

# ---------------- APP ----------------
app = FastAPI(title="JeelQuest RAG API", version="1.0")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MONGODB ATLAS ----------------
def get_db_connection():
    if not MONGO_URI:
        raise Exception("MONGO_URI not set")

    client = MongoClient(MONGO_URI)
    return client.get_default_database()

# ---------------- TEXT CLEANING ----------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"[¢©®«#]", "", text)
    text = re.sub(r"\b[eJo]\b", "", text)
    text = re.sub(r"^\s*o\s+", "- ", text, flags=re.MULTILINE)
    text = text.replace("&", "and")
    text = re.sub(r'(?<!\S)@(?!\S)', 'at', text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = "\n".join([line.strip() for line in text.splitlines()])
    return text.strip()

# ---------------- CHUNKING ----------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------- PDF EXTRACTION (NEW) ----------------
def extract_pdf_text(file_path: str) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

# ---------------- UPLOAD PDF ----------------
@app.post("/upload-documents/")
async def upload_files(documents: List[UploadFile] = File(...)):

    db = get_db_connection()
    documents_collection = db["documents"]

    uploaded_files = []

    for file in documents:

        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # ✅ PDF extraction FIXED
            raw_text = extract_pdf_text(file_path)

            extracted_text = clean_text(raw_text)
            chunks = split_text(extracted_text)

            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": file.filename}
                )
                for chunk in chunks if chunk.strip()
            ]

            if docs:
                vectorstore.add_documents(docs)

            documents_collection.insert_one({
                "filename": file.filename,
                "content": extracted_text,
                "chunks": chunks,
                "created_at": datetime.utcnow()
            })

            uploaded_files.append(file.filename)

            # cleanup (important cloud)
            os.remove(file_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Uploaded successfully",
        "files": uploaded_files
    }

# ---------------- CHATBOT ----------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):

    try:
        db = get_db_connection()
        chat_collection = db["chat_history"]

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke(request.query)

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        sources = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in source_docs
        ]

        chat_collection.insert_one({
            "query": request.query,
            "answer": answer,
            "sources": sources,
            "created_at": datetime.utcnow()
        })

        return {
            "query": request.query,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))