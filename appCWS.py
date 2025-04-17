from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os

import shutil
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from typing import List
import uvicorn
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from chromadb import ChromaDB  # Assuming ChromaDB is the library you're using for the database

app = FastAPI()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """تحميل الملف"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل في رفع الملف: {str(e)}")
    
    return {"message": "تم رفع الملف بنجاح", "file_path": file_path}

@app.post("/split/")
async def split_file(
    file_name: str = Form(...), 
    chunk_size: int = Form(...), 
    overlap: int = Form(...),
    split_type: str = Form("character")  # "character", "space" أو "semantic"
):
    """تقسيم الملف إلى أجزاء وتحويله إلى Excel"""
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    
    text = read_file_content(file_path)
    
    if split_type == "character":
        chunks = split_text_by_characters(text, chunk_size, overlap)
        output_file = os.path.splitext(file_path)[0] + "_ch_chunks.xlsx"
    elif split_type == "space":
        chunks = split_text_by_spaces(text, chunk_size, overlap)
        output_file = os.path.splitext(file_path)[0] + "_space_chunks.xlsx"
    elif split_type == "semantic":
        chunks = split_text_semantically(text)
        output_file = os.path.splitext(file_path)[0] + "_semantic_chunks.xlsx"
    else:
        raise HTTPException(status_code=400, detail="نوع التقسيم غير صحيح. استخدم 'character' أو 'space' أو 'semantic'.")
    
    pd.DataFrame({"Chunk": chunks}).to_excel(output_file, index=False)
    
    return {"message": "تمت معالجة الملف بنجاح", "output_file": output_file}

@app.post("/embed_chunks/")
async def embed_chunks(file_path: str = "reem1_space_chunks.xlsx"):
    """Embed chunks from an Excel file and save them in Chroma DB"""
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Check if 'Chunk' column exists
    if 'Chunk' not in df.columns:
        raise HTTPException(status_code=400, detail="Column 'Chunk' not found in the Excel file.")
    
    # Extract chunks
    chunks = df['Chunk'].tolist()
    
    # Load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Embed the chunks
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    # Initialize Chroma DB
    chroma_db = ChromaDB()  # Initialize your Chroma DB instance here
    
    # Save embeddings to Chroma DB
    for chunk, embedding in zip(chunks, embeddings):
        chroma_db.save_embedding(chunk, embedding)  # Assuming save_embedding is the method to save in Chroma DB
    
    return {"message": "Chunks embedded and saved successfully in Chroma DB"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
