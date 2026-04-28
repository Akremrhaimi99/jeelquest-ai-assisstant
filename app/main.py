import os
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient
from pymilvus import connections, Collection, utility

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

# Import pour l'authentification Google
import google.generativeai as genai

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Pour le chatbot
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")  # Pour les embeddings

# Configuration pour les embeddings (API Key 2)
genai.configure(api_key=GOOGLE_API_KEY2)

# Initialisation du modèle d'embedding avec GOOGLE_API_KEY2
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",  # Dimension 3072
    google_api_key=GOOGLE_API_KEY2  # Clé dédiée pour les embeddings
)

# Vérification et recréation de la collection Milvus avec la bonne dimension
def setup_vectorstore():
    """Crée ou recrée la collection Milvus avec la bonne dimension"""
    try:
        # Connexion à Milvus/Zilliz
        connections.connect(
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN
        )
        
        # Test pour connaître la dimension des embeddings
        test_vector = embedding_model.embed_query("test")
        expected_dim = len(test_vector)
        print(f"Dimension des embeddings générés : {expected_dim}")
        
        # Vérifier si la collection existe
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            collection.load()
            
            # Obtenir la dimension actuelle
            schema = collection.schema
            vector_field = None
            for field in schema.fields:
                if field.name == "vector" or field.dtype == 101:  # 101 = FloatVector
                    vector_field = field
                    break
            
            if vector_field:
                current_dim = vector_field.params.get('dim', 0)
                print(f"Dimension actuelle dans Milvus : {current_dim}")
                
                # Si les dimensions ne correspondent pas, supprimer la collection
                if current_dim != expected_dim:
                    print(f"⚠️ Incompatibilité de dimension : {current_dim} vs {expected_dim}")
                    print(f"Suppression de l'ancienne collection : {COLLECTION_NAME}")
                    collection.drop()
                    utility.drop_collection(COLLECTION_NAME)
                    print("✅ Ancienne collection supprimée")
            else:
                print("Champ vectoriel non trouvé dans le schéma")
        
        # Créer le vectorstore (créera automatiquement la collection si elle n'existe pas)
        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": ZILLIZ_URI,
                "token": ZILLIZ_TOKEN
            },
            auto_id=True
        )
        
        print(f"✅ Vectorstore initialisé avec la collection : {COLLECTION_NAME}")
        return vectorstore
        
    except Exception as e:
        print(f"Erreur lors de la configuration du vectorstore : {e}")
        # En cas d'erreur, essayer de créer directement
        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": ZILLIZ_URI,
                "token": ZILLIZ_TOKEN
            },
            auto_id=True
        )
        return vectorstore

# Initialisation du vectorstore
vectorstore = setup_vectorstore()

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


def split_text(text, chunk_size=300, overlap=50):
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

    # Vérifier que le vectorstore est accessible
    global vectorstore
    if vectorstore is None:
        vectorstore = setup_vectorstore()

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
            
            print(f"Nombre de chunks générés : {len(chunks)}")

            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": file.filename}
                )
                for chunk in chunks if chunk.strip()
            ]

            if docs:
                print(f"Ajout de {len(docs)} documents au vectorstore...")
                vectorstore.add_documents(docs)
                print("✅ Documents ajoutés avec succès")
            else:
                print("⚠️ Aucun document à ajouter")

            doc_entry = {
                "filename": file.filename,
                "filepath": file_path,
                "content": extracted_text,
                "chunks": chunks,
                "created_at": datetime.utcnow(),
            }

            documents_collection.insert_one(doc_entry)
            uploaded_files.append(file.filename)

            print(f"=== OK Document traité : {file.filename} ===")

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
# Recréer le retriever à partir du vectorstore mis à jour
def get_retriever():
    return vectorstore.as_retriever(search_kwargs={"k": 4})

class ChatRequest(BaseModel):
    query: str


@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        db = get_db_connection()
        chat_collection = db["chat_history"]

        # Utiliser GOOGLE_API_KEY pour le chatbot (clé dédiée)
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",  # Modèle valide
            google_api_key=GOOGLE_API_KEY,  # Clé dédiée pour le chat
            temperature=0
        )

        retriever = get_retriever()
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
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
