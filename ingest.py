import os
import fitz  # pymupdf
import pdfplumber

def extract_text_pymupdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def smart_extract(pdf_path):
    """
    Try PyMuPDF first.
    Fallback to pdfplumber if text is too short.
    """
    text = extract_text_pymupdf(pdf_path)

    if len(text.strip()) < 50:
        text = extract_text_pdfplumber(pdf_path)

    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks