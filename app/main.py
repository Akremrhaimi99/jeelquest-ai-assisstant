import os
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient

import fitz  # PyMuPDF
import re
import unicodedata
from dotenv import load_dotenv

from langchain_community.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel


load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
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

app = FastAPI(title="JeelQuest Questy V1", version="1.0")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- MONGODB ----------------
def get_db_connection():
    if not MONGO_URI:
        raise Exception("MONGO_URI not set")
    client = MongoClient(MONGO_URI)
    return client["documents_db"]


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


# ---------------- FILENAME CLEANING ----------------
def clean_filename(filename):
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^\w\.-]", "", filename)
    return filename


def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------- UPLOAD ----------------
@app.post("/upload-documents/")
async def upload_files(document: UploadFile = File(...)):

    uploaded_files = []
    db = get_db_connection()
    documents_collection = db["documents"]

    for file in [document]:

        file_path = None

        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        try:
            safe_filename = clean_filename(file.filename)
            file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if not os.path.exists(file_path):
                raise Exception(f"File not saved correctly: {file_path}")

            print("Saved file at:", file_path)

            doc = fitz.open(file_path)

            raw_text = ""
            for page in doc:
                raw_text += page.get_text("text") + "\n"

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

            doc_entry = {
                "filename": file.filename,
                "filepath": file_path,
                "content": extracted_text,
                "chunks": chunks,
                "created_at": datetime.utcnow(),
            }

            documents_collection.insert_one(doc_entry)
            uploaded_files.append(file.filename)

            print("=== OK ===", file.filename)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    return {
        "message": "Files uploaded and processed successfully",
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

        result = qa_chain.invoke({"query": request.query})

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        sources_text = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in source_docs
        ]

        chat_entry = {
            "query": request.query,
            "answer": answer,
            "num_sources": len(source_docs),
            "sources": sources_text,
            "created_at": datetime.utcnow()
        }

        chat_collection.insert_one(chat_entry)

        return {
            "query": request.query,
            "answer": answer,
            "sources": sources_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
