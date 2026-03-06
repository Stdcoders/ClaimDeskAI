"""
pipeline.py — Whisper + BERT classifiers + RAG chained together
================================================================
Loaded once at startup, reused for every call.
"""

import os
import sys
import json
import torch
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

INTENT_MODEL_PATH    = os.getenv("INTENT_MODEL_PATH",    r"D:\models\intent")
URGENCY_MODEL_PATH   = os.getenv("URGENCY_MODEL_PATH",   r"D:\models\urgency")
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", r"D:\models\sentiment")
RAG_FILE             = r"C:\Users\SRINIDHI\OneDrive\Desktop\NLP-Sem6\Claim-Dashboard\2_RAG.py"

# ── Load models once ──────────────────────────────────────────────────────────

_whisper      = None
_intent_mdl   = None
_intent_tok   = None
_urgency_mdl  = None
_urgency_tok  = None
_senti_mdl    = None
_senti_tok    = None
_intent_labels   = {}
_urgency_labels  = {}
_senti_labels    = {}
_rag_answer      = None


def load_all():
    global _whisper, _intent_mdl, _intent_tok, _urgency_mdl, _urgency_tok
    global _senti_mdl, _senti_tok, _intent_labels, _urgency_labels, _senti_labels
    global _rag_answer

    print("Loading Whisper...")
    from faster_whisper import WhisperModel
    _whisper = WhisperModel("base", device="cpu")

    print("Loading classifiers...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _intent_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _intent_mdl = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
    _intent_mdl.eval()

    _urgency_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _urgency_mdl = AutoModelForSequenceClassification.from_pretrained(URGENCY_MODEL_PATH)
    _urgency_mdl.eval()

    _senti_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _senti_mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    _senti_mdl.eval()

    # Load label maps
    with open(os.path.join(INTENT_MODEL_PATH,    "label_map.json")) as f:
        _intent_labels = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(URGENCY_MODEL_PATH,   "label_map.json")) as f:
        _urgency_labels = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(SENTIMENT_MODEL_PATH, "label_map.json")) as f:
        _senti_labels = {int(k): v for k, v in json.load(f).items()}

    print("Loading RAG...")
    if Path(RAG_FILE).exists():
        spec = importlib.util.spec_from_file_location("rag", RAG_FILE)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _rag_answer = mod.answer
    else:
        print(f"⚠ RAG file not found at {RAG_FILE} — RAG disabled")
        _rag_answer = lambda q, **kw: ("RAG not available", [])

    print("✅ Pipeline ready.")


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper."""
    if _whisper is None:
        raise RuntimeError("Pipeline not loaded. Call load_all() first.")
    segments, _ = _whisper.transcribe(audio_path, language="en")
    return " ".join(s.text for s in segments).strip()


def transcribe_bytes(audio_bytes: bytes, tmp_path: str = "/tmp/call_audio.wav") -> str:
    """Write bytes to temp file and transcribe."""
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    return transcribe_audio(tmp_path)


# ── Classification ────────────────────────────────────────────────────────────

def _classify(text: str, model, tokenizer, label_dict: dict, threshold=0.30):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs      = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    pred_idx   = torch.argmax(probs, dim=1).item()
    if confidence < threshold:
        return "Uncertain", confidence
    return label_dict[pred_idx], confidence


def classify_text(text: str) -> dict:
    """Run all 3 classifiers on text. Returns dict of results."""
    intent,    ci = _classify(text, _intent_mdl,  _intent_tok,  _intent_labels)
    urgency,   cu = _classify(text, _urgency_mdl, _urgency_tok, _urgency_labels)
    sentiment, cs = _classify(text, _senti_mdl,   _senti_tok,   _senti_labels)

    # Keyword overrides
    text_lower = text.lower()
    emergency_kw = ["icu", "emergency", "surgery", "critical", "life threatening"]
    if any(kw in text_lower for kw in emergency_kw) and urgency not in ("Critical", "High"):
        urgency = "High"
        cu = 1.0

    return {
        "intent":       intent,    "conf_intent":    round(ci, 3),
        "urgency":      urgency,   "conf_urgency":   round(cu, 3),
        "sentiment":    sentiment, "conf_sentiment": round(cs, 3),
    }


# ── RAG ───────────────────────────────────────────────────────────────────────

def get_rag_answer(query: str, intent: str, chat_history: list = None):
    """
    Run RAG with intent-enriched query.
    Returns (answer_text, sources_list).
    """
    if _rag_answer is None:
        return "Knowledge base not available.", []

    enriched = f"{query} (intent: {intent.replace('_', ' ')})"
    try:
        answer, sources = _rag_answer(enriched, chat_history=chat_history or [])
        return answer, sources
    except Exception as e:
        print(f"RAG error: {e}")
        return f"Unable to retrieve answer: {e}", []


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(audio_path: str, chat_history: list = None) -> dict:
    """
    Full pipeline: audio → transcript → classify → RAG.
    Returns complete result dict for storing in DB and sending to frontend.
    """
    # Step 1: Transcribe
    transcript = transcribe_audio(audio_path)
    if not transcript or len(transcript.split()) < 3:
        return {"error": "Could not transcribe audio", "transcript": transcript}

    # Step 2: Classify
    classifications = classify_text(transcript)

    # Step 3: RAG
    rag_text, sources = get_rag_answer(
        transcript,
        classifications["intent"],
        chat_history
    )

    # Step 4: Format sources for frontend
    sources_clean = [
        {
            "source_pdf": s["metadata"].get("source_pdf", ""),
            "category":   s["metadata"].get("category", ""),
            "score":      s.get("similarity", 0),
            "text":       s["text"][:300],
        }
        for s in (sources or [])
    ]

    return {
        "transcript":    transcript,
        "rag_answer":    rag_text,
        "sources":       sources_clean,
        **classifications,
    }