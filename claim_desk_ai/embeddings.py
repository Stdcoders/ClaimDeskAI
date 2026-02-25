import os
import json
import re
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# ==============================
# CONFIG
# ==============================

CHUNKS_FOLDER = r"C:\Users\SRINIDHI\OneDrive\Desktop\NLP-Sem6\NLP-SCE\Chunks"  # folder containing all json files
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


# ==============================
# LOAD MODEL
# ==============================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.\n")


# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):
    if not text:
        return ""

    text = text.encode("utf-8", "ignore").decode()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==============================
# NORMALIZE CHUNK STRUCTURE
# ==============================

def normalize_chunk(chunk, source_file):

    normalized = {
        "content": "",
        "metadata": {}
    }

    # Extract content safely
    normalized["content"] = clean_text(chunk.get("content", ""))

    # Case 1: Already has metadata
    if "metadata" in chunk and isinstance(chunk["metadata"], dict):
        normalized["metadata"] = chunk["metadata"]

    # Case 2: Flat structure (like FAQs)
    else:
        normalized["metadata"] = {
            "source": chunk.get("source", source_file),
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_number": chunk.get("chunk_number", "")
        }

    # Always attach filename
    normalized["metadata"]["file_name"] = source_file

    # Add universal signals
    text = normalized["content"]
    normalized["metadata"]["word_count"] = len(text.split())
    normalized["metadata"]["contains_number"] = bool(re.search(r'\d+', text))
    normalized["metadata"]["contains_percentage"] = bool(re.search(r'\d+\.?\d*%', text))

    return normalized


# ==============================
# LOAD ALL FILES
# ==============================

all_chunks = []

print("Loading chunk files...\n")

for file in os.listdir(CHUNKS_FOLDER):

    if not file.endswith(".json"):
        continue

    file_path = os.path.join(CHUNKS_FOLDER, file)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if not isinstance(data, list):
            continue

        for chunk in data:
            normalized = normalize_chunk(chunk, file)
            if normalized["content"]:
                all_chunks.append(normalized)

print(f"Total chunks loaded: {len(all_chunks)}\n")


# ==============================
# PREPARE TEXT FOR EMBEDDING
# ==============================

processed_texts = []

for chunk in all_chunks:

    meta = chunk["metadata"]

    prefix = f"""
    File: {meta.get('file_name', '')}
    Source: {meta.get('source', '')}
    Clause: {meta.get('clause_number', '')}

    Content:
    """

    processed_texts.append(prefix + chunk["content"])


# ==============================
# GENERATE EMBEDDINGS
# ==============================

print("Generating embeddings...")
embeddings = model.encode(
    processed_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Embeddings generated.\n")


# ==============================
# BUILD FAISS INDEX
# ==============================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.\n")


# ==============================
# SAVE
# ==============================

faiss.write_index(index, INDEX_FILE)

with open(METADATA_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("Vector database ready 🚀")