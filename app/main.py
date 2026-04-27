import os
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient

from unstructured.partition.pdf import partition_pdf
import re
import unicodedata
from dotenv import load_dotenv

from langchain_community.vectorstores import Milvus

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.schema import Document

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


from pydantic import BaseModel


load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("HUGGINGFACE_API_KEY")

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=api_key,
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


app = FastAPI(title="JeelQuest Questy V1", version="1.0")

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
    
    # Normaliser Unicode (supprime les caractères bizarres) 
    text = unicodedata.normalize("NFKC", text) 
    
    # Remplacer les caractères invisibles et tabulations exemple "Hello\xa0World" 
    text = text.replace("\xa0", " ").replace("\t", " ") 
    
    # Supprimer les caractères spéciaux typiques du PDF 
    text = re.sub(r"[¢©®«#]", "", text) 
    
    # Supprimer les lettres seules "e" et "J" qui apparaissent souvent comme bruit (exemple "." ce transforme en "e" et "➡" en "J" ) 
    text = re.sub(r"\b[eJo]\b", "", text) 
    
    # de "o" en "- " pour les listes 
    text = re.sub(r"^\s*o\s+", "- ", text, flags=re.MULTILINE) 
    
    # Remplacer & par and 
    text = text.replace("&", "and") 
    
    # Remplacer @ par at (mais pas les emails) 
    text = re.sub(r'(?<!\S)@(?!\S)', 'at', text) 
    
    # Supprimer sauts de ligne multiples 
    text = re.sub(r"\n+", "\n", text) 
    
    # Supprimer espaces multiples 
    text = re.sub(r"[ ]{2,}", " ", text) 
    
    # Corriger les mots coupés (ex: "exem-\nple" → "exemple") 
    text = re.sub(r"-\s*\n\s*", "", text) 
    
    # Supprimer espaces en début / fin de ligne 
    text = "\n".join([line.strip() for line in text.splitlines()]) 
    
    return text.strip()


# ---------------- Filename CLEANING ----------------
def clean_filename(filename):
    # Normaliser Unicode
    filename = unicodedata.normalize("NFKD", filename)

    # Supprimer caractères non ASCII
    filename = filename.encode("ascii", "ignore").decode("ascii")

    # Remplacer espaces par _
    filename = filename.replace(" ", "_")

    # Supprimer caractères dangereux
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


@app.post("/upload-documents/")
async def upload_files(documents: List[UploadFile] = File(...)):

    uploaded_files = []
    db = get_db_connection()
    documents_collection = db["documents"]

    for file in documents:

        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        try:
            
            # Nettoyer le nom du fichier
            safe_filename = clean_filename(file.filename)
            
            # Chemin ABSOLU (important pour pdf2image car unstructured travaille meme avec ocr pour les fichier scaner) pour le chemain complet du fichier avec c:\ 
            file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))
            
            # Sauvegarde
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

             #  Vérification (très utile debug)
            if not os.path.exists(file_path):
                raise Exception(f"File not saved correctly: {file_path}")

            print("Saved file at:", file_path)


            # Extraction
            elements = partition_pdf(
    		filename=file_path,
    		strategy="fast"
	    )		
            raw_text = "\n".join(
                [el.text for el in elements if hasattr(el, "text") and el.text]
            )

            # Nettoyage
            extracted_text = clean_text(raw_text)

            # Chunking
            chunks = split_text(extracted_text)

            # Convertir en document pour Milvus
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": file.filename}
                )
                for chunk in chunks if chunk.strip()
            ]

            # Stockage sur Milvus
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
	        # Nettoyage fichier après usage
	        os.remove(file_path)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Files uploaded and processed successfully",
        "files": uploaded_files
    }


# ---------------- CHATBOT ----------------

# pour permet de faire des recherches similaires dans Milvus avec les 4 chunks les plus pertinents
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# query il faut etre une chaine de caractères (string) et c'est obligatoire
class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:

        db = get_db_connection()
        chat_collection = db["chat_history"]

        # récupérer API key Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")

        # initialiser de LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=google_api_key,
            temperature=0
        )

        # créer le pipeline RAG
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # poser une question
        result = qa_chain.invoke(request.query)

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        # formatter sources de texte pour la réponse (afficher les 300 premiers caractères et la source)
        sources_text = []
        for doc in source_docs:
            sources_text.append({
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown")
            })

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
