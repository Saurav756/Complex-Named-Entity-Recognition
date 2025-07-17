import os
import PyPDF2
from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf_file(file_path: str) -> str:
    pdf = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def load_document(file_path: str) -> str:
    if file_path.endswith(".txt"):
        return read_text_file(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type")

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def load_and_split(file_path: str) -> List[str]:
    raw_text = load_document(file_path)
    return split_text(raw_text)

if __name__ == "__main__":
    file_path = "os1a-slides.pdf"  # replace with your path
    chunks = load_and_split(file_path)
    print(f"Loaded {len(chunks)} chunks.")
    print(chunks[0])
