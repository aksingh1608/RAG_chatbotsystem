import os
import numpy as np
import faiss
import cohere
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
co = cohere.Client(os.getenv("COHERE_API_KEY"))

INDEX_PATH = "faiss_index"
EMBED_DIM = 384  # dimension of all-MiniLM-L6-v2


def create_index(chunks, doc_metadata=None, append=False):
    """
    Create or append to FAISS index.
    
    Args:
        chunks: List of text chunks to index
        doc_metadata: List of document metadata dicts
        append: If True, append to existing index; if False, create new
    """
    embeddings = embedder.encode(chunks)
    
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    if append and os.path.exists(f"{INDEX_PATH}/index.faiss"):
        # Load existing index and append
        index = faiss.read_index(f"{INDEX_PATH}/index.faiss")
        index.add(np.array(embeddings))
        
        # Append chunks
        with open(f"{INDEX_PATH}/chunks.txt", "a", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.replace("\n", " ") + "\n")
        
        # Append metadata
        existing_metadata = []
        if os.path.exists(f"{INDEX_PATH}/metadata.json"):
            with open(f"{INDEX_PATH}/metadata.json", "r") as f:
                existing_metadata = json.load(f)
        
        if doc_metadata:
            existing_metadata.extend(doc_metadata)
        
        with open(f"{INDEX_PATH}/metadata.json", "w") as f:
            json.dump(existing_metadata, f, indent=2)
    else:
        # Create new index
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(np.array(embeddings))
        
        # Write chunks
        with open(f"{INDEX_PATH}/chunks.txt", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.replace("\n", " ") + "\n")
        
        # Write metadata
        if doc_metadata:
            with open(f"{INDEX_PATH}/metadata.json", "w") as f:
                json.dump(doc_metadata, f, indent=2)
    
    # Save index
    faiss.write_index(index, f"{INDEX_PATH}/index.faiss")
    
    # Update stats
    metadata_count = 0
    if os.path.exists(f"{INDEX_PATH}/metadata.json"):
        with open(f"{INDEX_PATH}/metadata.json", "r") as f:
            metadata = json.load(f)
            metadata_count = len(metadata)
    
    stats = {
        'total_chunks': index.ntotal,
        'total_documents': metadata_count,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{INDEX_PATH}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Index {'updated' if append else 'created'} successfully.")


def load_index():
    """Load FAISS index and chunks from disk"""
    index = faiss.read_index(f"{INDEX_PATH}/index.faiss")

    with open(f"{INDEX_PATH}/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.readlines()

    return index, chunks


def retrieve(query, index, chunks, top_k=3):
    """Retrieve top-k most relevant chunks for a query"""
    q_embed = embedder.encode([query])
    distances, indices = index.search(np.array(q_embed), top_k)

    results = [chunks[i] for i in indices[0]]
    return results


def generate_answer(query, context_chunks):
    """Generate answer using Cohere API with retrieved context"""
    context = "\n".join(context_chunks)

    response = co.chat(
        model="command-a-03-2025",
        message=query,
        preamble=(
            "You are a helpful assistant. "
            "Answer strictly using the provided context. "
            "If the answer is not in the context, say 'Not found in document.'"
        ),
        documents=[{"text": context}]
    )

    return response.text


def get_index_stats():
    """Get statistics about the current index"""
    if os.path.exists(f"{INDEX_PATH}/stats.json"):
        with open(f"{INDEX_PATH}/stats.json", "r") as f:
            return json.load(f)
    return {
        'total_chunks': 0,
        'total_documents': 0,
        'last_updated': 'N/A'
    }


def list_indexed_documents():
    """List all indexed documents with metadata"""
    if os.path.exists(f"{INDEX_PATH}/metadata.json"):
        with open(f"{INDEX_PATH}/metadata.json", "r") as f:
            return json.load(f)
    return []
