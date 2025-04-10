# rag-sys
Document processing API for splitting text files into overlapping chunks

rag-sys is a FastAPI application designed to process various document formats (TXT, PDF, DOCX) and split their content into manageable chunks with configurable overlap sizes. The system provides a RESTful API for uploading documents and generating Excel outputs containing the processed text segments.

## Features
- Support for TXT, PDF, and DOCX file formats
- Configurable chunk size and overlap parameters
- Automatic Excel output generation
- Clean API endpoints for easy integration

## Requirements
- Python 3.7+
- FastAPI
- Uvicorn
- PyPDF2
- python-docx
- pandas

## Installation
```bash
python -m venv venv
source venv/bin/activate #.\venv\scripts\activate

pip install fastapi uvicorn PyPDF2 python-docx pandas 
```

## API Endpoints
### Upload File
Upload a document file to the server
```http
POST /upload/
```
Request Body:
- file: File (TXT/PDF/DOCX)

Example Response:
```json
{
    "message": "File uploaded successfully",
    "file_path": "/uploaded_files/document.pdf"
}
```

### Split Document
Split the uploaded document into chunks
```http
POST /split/
```
Form Parameters:
- file_name: string (name of uploaded file)
- chunk_size: integer (number of characters per chunk)
- overlap: integer (overlap size between chunks)

Example Response:
```json
{
    "message": "File processed successfully",
    "output_file": "/uploaded_files/document_chunks.xlsx"
}
```

## Usage Example
Using curl:
```bash

curl -X POST http://localhost:8000/upload/ \
     -F "file=@document.pdf"


curl -X POST http://localhost:8000/split/ \
     -F "file_name=document.pdf" \
     -F "chunk_size=1000" \
     -F "overlap=200" \
     -F "split_type=character or space"


```

## Development Server
Run the development server:
```bash
python app.py
```
