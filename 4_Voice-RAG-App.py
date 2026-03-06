"""
4_voice_rag_app.py  —  Integrated Voice + RAG Medical Assistant
================================================================
Combines:
  - bertwithvoice.py  (Whisper + DistilBERT intent/urgency/sentiment)
  - 2_rag.py          (ChromaDB hybrid search + HF LLM answer)

USAGE:
  streamlit run 4_voice_rag_app.py

INSTALL:
  pip install sounddevice scipy faster-whisper transformers torch
              pyttsx3 chromadb sentence-transformers huggingface_hub
              rank_bm25 streamlit numpy
"""

import os
import sys
import importlib.util
import threading
import json
import numpy as np
import torch
import streamlit as st

# ── Load 2_rag.py from same folder ────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "rag", os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_rag.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
rag_answer   = _mod.answer
expand_query = _mod.expand_query

# ── Model paths — UPDATE THESE ────────────────────────────────────────────────
INTENT_MODEL_PATH    = r"D:\models\intent"
URGENCY_MODEL_PATH   = r"D:\models\urgency"
SENTIMENT_MODEL_PATH = r"D:\models\sentiment"
AUDIO_FILE           = "live_audio.wav"
RECORD_DURATION      = 6    # seconds
SAMPLE_RATE          = 16000
# ──────────────────────────────────────────────────────────────────────────────

with open(r"D:\models\intent\label_map.json") as f:
    intent_labels = {int(k): v for k, v in json.load(f).items()}

with open(r"D:\models\urgency\label_map.json") as f:
    urgency_labels = {int(k): v for k, v in json.load(f).items()}

with open(r"D:\models\sentiment\label_map.json") as f:
    sentiment_labels = {int(k): v for k, v in json.load(f).items()}

SLA_MAP = {
    "Critical": "within 2 hours",
    "High":     "within 6 hours",
    "Medium":   "within 24 hours",
    "Low":      "within 48 hours",
}

INTENT_ICONS = {
    "Claim_Status_Query":   "📋",
    "Complaint":            "⚠️",
    "Coverage_Query":       "🛡️",
    "Dispute_Clarification":"⚖️",
    "Document_Query":       "📄",
    "Escalation_Request":   "🚨",
    "Policy_Query":         "📜",
    "Positive_Feedback":    "👍",
    "Process_Clarification":"🔄",
    "Reimbursement_Query":  "💰",
}
URGENCY_COLORS   = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#22c55e"}
SENTIMENT_COLORS = {"Negative": "#ef4444", "Neutral": "#6b7280",  "Positive": "#22c55e"}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — load once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Whisper transcription model...")
def load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu")


@st.cache_resource(show_spinner="Loading classification models...")
def load_classifiers():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    intent_tok  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    urgency_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    senti_tok   = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    intent_mdl  = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
    urgency_mdl = AutoModelForSequenceClassification.from_pretrained(URGENCY_MODEL_PATH)
    senti_mdl   = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)

    return (intent_tok, intent_mdl), (urgency_tok, urgency_mdl), (senti_tok, senti_mdl)


@st.cache_resource(show_spinner="Loading TTS engine...")
def load_tts():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def record_audio(duration=RECORD_DURATION, fs=SAMPLE_RATE) -> bool:
    """Record audio from microphone, save to AUDIO_FILE. Returns True on success."""
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        wav.write(AUDIO_FILE, fs, (recording * 32767).astype(np.int16))
        return True
    except Exception as e:
        st.error(f"Recording error: {e}")
        return False


def transcribe(whisper) -> str:
    """Transcribe AUDIO_FILE using Whisper."""
    segments, _ = whisper.transcribe(AUDIO_FILE, language="en")
    return " ".join(s.text for s in segments).strip()


def predict(text, model, tokenizer, label_dict, threshold=0.30):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities   = torch.softmax(outputs.logits, dim=1)
    confidence      = torch.max(probabilities).item()
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if confidence < threshold:
        return "Uncertain", confidence

    return label_dict[predicted_class], confidence


def speak_async(engine, text: str):
    """Speak text in a background thread so Streamlit doesn't block."""
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


def build_sla_response(intent: str, urgency: str) -> str:
    sla = SLA_MAP.get(urgency, "within 48 hours")
    return f"Your {intent.replace('_', ' ')} will be resolved {sla}."


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedClaim Voice Assistant",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #0f172a;
    margin-bottom: 0;
    line-height: 1.1;
}
.main-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 300;
}

/* Chat bubbles */
.bubble-user {
    background: #0f172a;
    color: #f8fafc;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.5;
}
.bubble-bot {
    background: #f1f5f9;
    color: #0f172a;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.6;
    border-left: 3px solid #3b82f6;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 3px;
    letter-spacing: 0.03em;
}

/* Metric cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
}

/* Record button */
.stButton > button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: #1e3a5f !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(15,23,42,0.3) !important;
}

.divider { border-top: 1px solid #e2e8f0; margin: 16px 0; }
.transcript-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 10px 14px;
    font-style: italic;
    color: #92400e;
    font-size: 0.9rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("chat_history",  []),
    ("last_intent",   None),
    ("last_urgency",  None),
    ("last_sentiment",None),
    ("last_sources",  []),
    ("last_conf",     {}),
    ("recording",     False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-title">🏥 MedClaim Voice Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Speak your medical claim query — voice-powered RAG with intent analysis</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    hf_token = st.text_input("HuggingFace Token", type="password", placeholder="hf_...")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.divider()
    record_duration = st.slider("Recording duration (seconds)", 3, 15, RECORD_DURATION)
    speak_response  = st.toggle("🔊 Speak responses aloud", value=True)
    show_sources    = st.toggle("📄 Show RAG sources",      value=True)
    show_expansion  = st.toggle("🔎 Show query expansion",  value=False)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history  = []
        st.session_state.last_intent   = None
        st.session_state.last_urgency  = None
        st.session_state.last_sentiment= None
        st.session_state.last_sources  = []
        st.rerun()

    st.divider()
    st.caption("🎤 Whisper base (CPU)")
    st.caption("🧠 DistilBERT (intent/urgency/sentiment)")
    st.caption("🔍 S-PubMedBert + BM25 hybrid search")
    st.caption("🤖 Zephyr-7B via HF API")

# Main columns
col_chat, col_analysis = st.columns([3, 1.2], gap="large")

# ── RIGHT PANEL: Analysis ─────────────────────────────────────────────────────
with col_analysis:
    st.markdown("### 📊 Analysis")

    if st.session_state.last_intent:
        intent   = st.session_state.last_intent
        urgency  = st.session_state.last_urgency
        sentiment= st.session_state.last_sentiment
        confs    = st.session_state.last_conf

        icon = INTENT_ICONS.get(intent, "🔹")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Intent</div>
            <div class="metric-value">{icon} {intent.replace('_',' ')}</div>
            <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px">
                confidence {confs.get('intent', 0):.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        urg_color = URGENCY_COLORS.get(urgency, "#6b7280")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Urgency</div>
            <div class="metric-value" style="color:{urg_color}">⏱ {urgency}</div>
            <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px">
                SLA: {SLA_MAP.get(urgency,'within 48 hours')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        sent_color = SENTIMENT_COLORS.get(sentiment, "#6b7280")
        sent_emoji = {"Positive":"😊","Neutral":"😐","Negative":"😟"}.get(sentiment,"")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentiment</div>
            <div class="metric-value" style="color:{sent_color}">
                {sent_emoji} {sentiment}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_sources and st.session_state.last_sources:
            st.markdown("**📄 Sources used**")
            for i, c in enumerate(st.session_state.last_sources, 1):
                m = c["metadata"]
                with st.expander(f"[{i}] {m['source_pdf'][:30]}...", expanded=False):
                    st.caption(f"Category: {m['category']}")
                    st.caption(f"Score: {c['similarity']}")
                    st.text(c["text"][:300] + "...")
    else:
        st.markdown("""
        <div class="metric-card" style="text-align:center;color:#94a3b8;padding:30px 16px;">
            <div style="font-size:2rem">🎤</div>
            <div style="font-size:0.85rem;margin-top:8px">
                Press Record to analyse your first query
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── LEFT PANEL: Chat ──────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Conversation")

    # Chat display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;color:#94a3b8;padding:40px 0;">
                <div style="font-size:3rem">🎙️</div>
                <div style="margin-top:12px;font-size:0.95rem">
                    Your conversation will appear here.<br>Press the button below to start.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for turn in st.session_state.chat_history:
                st.markdown(
                    f'<div class="bubble-user">🎤 {turn["user"]}</div>',
                    unsafe_allow_html=True
                )
                if turn.get("transcript_note"):
                    st.markdown(
                        f'<div class="transcript-box">📝 Heard: "{turn["transcript_note"]}"</div>',
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f'<div class="bubble-bot">{turn["bot"]}</div>',
                    unsafe_allow_html=True
                )
                # Show badges for this turn
                intent_i   = turn.get("intent", "")
                urgency_i  = turn.get("urgency", "")
                sentiment_i= turn.get("sentiment", "")
                uc = URGENCY_COLORS.get(urgency_i, "#6b7280")
                sc = SENTIMENT_COLORS.get(sentiment_i, "#6b7280")
                st.markdown(
                    f'<span class="badge" style="background:#e0e7ff;color:#3730a3">'
                    f'{INTENT_ICONS.get(intent_i,"🔹")} {intent_i.replace("_"," ")}</span>'
                    f'<span class="badge" style="background:{uc}22;color:{uc}">⏱ {urgency_i}</span>'
                    f'<span class="badge" style="background:{sc}22;color:{sc}">{sentiment_i}</span>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RECORD BUTTON ─────────────────────────────────────────────────────────
    if not os.environ.get("HF_TOKEN"):
        st.warning("⚠️ Enter your HuggingFace token in the sidebar to enable responses.")

    btn_label = "🎤  Record & Ask"
    if st.button(btn_label, use_container_width=True):

        if not os.environ.get("HF_TOKEN"):
            st.error("Please enter your HuggingFace token in the sidebar first.")
            st.stop()

        # Load models
        whisper = load_whisper()
        (intent_tok, intent_mdl), (urg_tok, urg_mdl), (senti_tok, senti_mdl) = load_classifiers()
        tts_engine = load_tts() if speak_response else None

        # Step 1: Record
        with st.status("🎤 Recording... speak now!", expanded=True) as status:
            st.write(f"Recording for {record_duration} seconds...")
            success = record_audio(duration=record_duration)
            if not success:
                status.update(label="❌ Recording failed", state="error")
                st.stop()

            # Step 2: Transcribe
            status.update(label="📝 Transcribing...")
            st.write("Transcribing audio...")
            transcript = transcribe(whisper)

            if not transcript:
                status.update(label="❌ Could not understand audio", state="error")
                st.warning("No speech detected. Please try again.")
                st.stop()

            st.write(f"**Heard:** {transcript}")

            if len(transcript.split()) < 3:
                status.update(label="⚠️ Too short", state="error")
                st.warning("Too short — please speak a full sentence.")
                st.stop()

            # Step 3: Classify
            status.update(label="🧠 Analysing intent, urgency, sentiment...")
            st.write("Running classification models...")

            intent,   conf_i = predict(transcript, intent_mdl,  intent_tok,  intent_labels)
            urgency,  conf_u = predict(transcript, urg_mdl,      urg_tok,     urgency_labels)
            sentiment,conf_s = predict(transcript, senti_mdl,    senti_tok,   sentiment_labels)

            st.write(f"Intent: **{intent}** ({conf_i:.0%}) | "
                     f"Urgency: **{urgency}** ({conf_u:.0%}) | "
                     f"Sentiment: **{sentiment}** ({conf_s:.0%})")

            # Step 4: RAG
            status.update(label="🔍 Searching knowledge base...")
            st.write("Retrieving relevant medical documents...")

            # Build history for multi-turn
            history = []
            for turn in st.session_state.chat_history[-4:]:
                history.append({"role": "user",      "content": turn["user"]})
                history.append({"role": "assistant",  "content": turn["bot"]})

            # Enrich query with intent context for better retrieval
            enriched_query = f"{transcript} (intent: {intent.replace('_',' ')})"

            try:
                rag_resp, sources = rag_answer(enriched_query, chat_history=history)
            except Exception as e:
                status.update(label="❌ RAG error", state="error")
                st.error(f"RAG error: {e}")
                st.stop()

            # Append SLA note to answer
            sla_note  = build_sla_response(intent, urgency)
            final_ans = f"{rag_resp}\n\n*{sla_note}*"

            status.update(label="✅ Done!", state="complete")

        # Step 5: Speak
        if speak_response and tts_engine:
            speak_async(tts_engine, rag_resp)

        # Step 6: Store in session
        expanded = expand_query(transcript)
        st.session_state.last_intent    = intent
        st.session_state.last_urgency   = urgency
        st.session_state.last_sentiment = sentiment
        st.session_state.last_sources   = sources
        st.session_state.last_conf      = {"intent": conf_i, "urgency": conf_u, "sentiment": conf_s}

        st.session_state.chat_history.append({
            "user"           : transcript,
            "bot"            : final_ans,
            "intent"         : intent,
            "urgency"        : urgency,
            "sentiment"      : sentiment,
            "transcript_note": transcript if show_expansion else None,
        })

        st.rerun()