"""
2_rag.py  —  Retrieval + Answer Generation (HuggingFace Inference API)
=======================================================================
Improvements over v1:
  - Medical-domain embedding model (S-PubMedBert-MS-MARCO)
  - Hybrid search: semantic + BM25 keyword (better recall)
  - Query expansion: expands short/abbreviation queries
  - TOP_K increased to 8

Requires: pip install chromadb sentence-transformers huggingface_hub rank_bm25

Standalone test:
  python 2_rag.py "What is the PM JAY scheme?"
"""

import sys
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = r"C:\Users\SRINIDHI\OneDrive\Desktop\NLP-Sem6\NLP-SCE\RAG\chroma_db"
COLLECTION_NAME = "medical_chunks"

# Must match exactly what was used in 1_ingest.py
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

LLM_MODEL      = "HuggingFaceH4/zephyr-7b-beta"
MAX_NEW_TOKENS = 512
TOP_K          = 8      # retrieve more candidates for better recall

SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant helping healthcare professionals "
    "understand medical guidelines, procedures, and treatment protocols. "
    "Answer using ONLY the context provided. Be clear and concise. "
    "If the context does not contain enough information, say so honestly."
)

# ── Query expansion dictionary ─────────────────────────────────────────────────
# Maps common short queries / abbreviations to richer search text
QUERY_EXPANSIONS = {
    "pm jay"        : "PM JAY Pradhan Mantri Jan Arogya Yojana health insurance scheme coverage",
    "pmjay"         : "PM JAY Pradhan Mantri Jan Arogya Yojana health insurance scheme coverage",
    "pmrssm"        : "Pradhan Mantri Rashtriya Swasthya Suraksha Mission health scheme",
    "claim"         : "claim settlement reimbursement procedure submission",
    "surgery"       : "surgical procedure operation treatment protocol",
    "package"       : "medical package cost procedure coverage",
    "grievance"     : "grievance redressal complaint resolution process",
    "empanelment"   : "hospital empanelment registration network criteria",
    "beneficiary"   : "beneficiary eligibility patient entitlement",
    "reimbursement" : "reimbursement claim payment procedure",
    "hbp"           : "health benefit package procedures covered",
    "stg"           : "standard treatment guidelines protocol",
}
# ──────────────────────────────────────────────────────────────────────────────

_collection = None


def get_collection():
    global _collection
    if _collection:
        return _collection
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    return _collection


def expand_query(query: str) -> str:
    """
    Expand short or abbreviated queries with domain-specific synonyms.
    Helps the embedding model find relevant chunks for terse inputs.
    """
    q_lower = query.lower()
    extras  = []
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in q_lower:
            extras.append(expansion)
    if extras:
        expanded = f"{query} {' '.join(extras)}"
        return expanded
    return query


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Pure semantic search — returns top_k chunks by cosine similarity."""
    results = get_collection().query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "metadata": m, "similarity": round(1 - s, 4)}
        for d, m, s in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Hybrid search: semantic (60%) + BM25 keyword (40%).

    Strategy:
      1. Expand query for better semantic coverage
      2. Retrieve 3x candidates via semantic search
      3. Re-rank candidates with BM25 on original query
      4. Combine normalised scores and return top_k
    """
    try:
        from rank_bm25 import BM25Okapi
        use_bm25 = True
    except ImportError:
        print("⚠ rank_bm25 not installed — falling back to semantic-only search.")
        print("  Install with:  pip install rank_bm25")
        use_bm25 = False

    # Step 1: Expand query and fetch extra candidates
    expanded_query    = expand_query(query)
    candidate_pool    = top_k * 3
    semantic_results  = retrieve(expanded_query, top_k=candidate_pool)

    if not use_bm25 or not semantic_results:
        return semantic_results[:top_k]

    # Step 2: BM25 on original (unexpanded) query over candidate texts
    tokenised_corpus  = [c["text"].lower().split() for c in semantic_results]
    bm25              = BM25Okapi(tokenised_corpus)
    bm25_scores       = bm25.get_scores(query.lower().split())

    # Step 3: Normalise both score sets to [0, 1]
    max_sem = max((c["similarity"] for c in semantic_results), default=1) or 1
    max_bm  = max(bm25_scores, default=1) or 1

    for i, c in enumerate(semantic_results):
        sem_norm  = c["similarity"] / max_sem
        bm25_norm = float(bm25_scores[i]) / max_bm
        # Weighted combination: 60% semantic, 40% keyword
        c["combined_score"] = round(0.6 * sem_norm + 0.4 * bm25_norm, 4)
        c["similarity"]     = c["combined_score"]   # reuse field for display

    # Step 4: Re-rank and trim
    semantic_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return semantic_results[:top_k]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(
            f"[Source {i}: {m['source_pdf']} | {m['category']} | "
            f"score: {c['similarity']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(query: str, chunks: list[dict], chat_history=None) -> list[dict]:
    """Build chat messages list for the HF Inference API."""
    context  = build_context(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include last 2 turns of history (4 messages) to keep prompt short
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}"
        ),
    })
    return messages


def answer(query: str, chat_history=None) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline:
      1. Hybrid retrieve (semantic + BM25) with query expansion
      2. Build chat messages with context
      3. Call HuggingFace Inference API
    Returns (answer_text, retrieved_chunks)
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not set.\n"
            "Enter it in the Streamlit sidebar or run:  set HF_TOKEN=hf_yourtoken"
        )

    from huggingface_hub import InferenceClient

    # Step 1: Retrieve
    chunks   = hybrid_retrieve(query)

    # Step 2: Build messages
    messages = build_messages(query, chunks, chat_history)

    # Step 3: Call API
    client   = InferenceClient(model=LLM_MODEL, token=hf_token)
    response = client.chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.3,
        top_p=0.9,
    )

    answer_text = response.choices[0].message.content.strip()
    return answer_text, chunks


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is the PM JAY scheme?"
    print(f"\nQuery     : {query}")
    print(f"Expanded  : {expand_query(query)}")
    print("=" * 60)

    resp, chunks = answer(query)

    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['similarity']}  {c['metadata']['source_pdf']}")
    print(f"\nANSWER:\n{resp}")