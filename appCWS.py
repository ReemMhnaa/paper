from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from typing import List
import uvicorn
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader


app = FastAPI()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# إعداد ChromaDB
CHROMA_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="document_chunks")


def read_file_content(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()
    text = ""
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == "pdf":
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == "docx":
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_file_content(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()
    text = ""
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == "pdf":
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == "docx":
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

def split_text_by_characters(text: str, chunk_size: int, overlap: int) -> List[str]:
    """تقسيم النص إلى أجزاء بناءً على عدد الأحرف"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator='')
    return text_splitter.split_text(text)

def split_text_by_spaces(text: str, chunk_size: int, overlap: int) -> List[str]:
    """تقسيم النص بناءً على المسافات"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

def split_text_semantically(text: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    """تقسيم النص بناءً على التشابه الدلالي مع تحديد العتبة ديناميكيًا"""

    # تحميل نموذج تحويل الجمل إلى متجهات
    model = SentenceTransformer(model_name)

    # تقسيم النص إلى جمل
    sentences = text.split(". ")
    if len(sentences) < 2:
        return sentences  # لا يمكن تطبيق التحليل على نص من جملة واحدة

    # حساب المتجهات لكل جملة
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # حساب التشابه بين الجمل المتتالية
    similarities = [
        np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        for i in range(len(sentences) - 1)
    ]

    # حساب العتبة الديناميكية: المتوسط - نصف الانحراف المعياري
    mean_similarity = np.mean(similarities)
    std_dev_similarity = np.std(similarities)
    threshold = mean_similarity - (0.5 * std_dev_similarity)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        if similarities[i - 1] > threshold:  # مقارنة التشابه بالعتبة الديناميكية
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks




@app.post("/embed_chunks/")
async def embed_chunks(file_path: str = Form(...)):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="الملف غير موجود")

    df = pd.read_excel(file_path)

    if 'Chunk' not in df.columns:
        raise HTTPException(status_code=400, detail="الملف لا يحتوي على عمود 'Chunk'.")

    chunks = df['Chunk'].dropna().tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(uuid.uuid4())]
        )

    return {"message": "تم حفظ التضمينات في ChromaDB بنجاح"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
