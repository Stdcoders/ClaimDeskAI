# ClaimDesk AI 🏥

> An intelligent agent assistance system for PMJAY health insurance claims processing — built with Whisper, DistilBERT, RAG, and Llama 3.3.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18.2-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow)

---
## Demo Video Link - https://drive.google.com/file/d/1xJ73EAfOvHzo_CdOtcXr21kX_9R4xsiN/view?usp=drive_link
## 📌 Overview

ClaimDesk AI is a real-time voice assistant that helps PMJAY (Pradhan Mantri Jan Arogya Yojana) health insurance call center agents handle customer queries efficiently. The system:

- **Transcribes** customer speech using OpenAI Whisper
- **Classifies** intent, urgency, and sentiment in parallel using fine-tuned DistilBERT models
- **Retrieves** accurate, grounded answers from official PMJAY policy documents using RAG
- **Auto-resolves** routine queries without human intervention
- **Generates** AI-powered resolution action plans for complex cases (complaints, escalations, reimbursements) using Llama 3.3 via Groq
- **Reads answers aloud** to the agent via browser TTS

---

## 🎯 Problem Statement

Design an AI-powered solution that will:
- Automatically schedule callbacks and follow-ups based on priority and urgency
- Autonomously handle routine tasks
- Analyze client sentiment on call
- Ensure agents have quick access to accurate information through a knowledge base

---

## 🏗️ System Architecture

```
Customer Voice
      │
      ▼
┌─────────────────┐
│  Whisper (tiny) │  ← Speech-to-Text (int8 quantized, CPU)
│  + int8 quant   │
└────────┬────────┘
         │ Transcript
         ▼
┌─────────────────────────────────────┐
│     DistilBERT Classifiers (×3)     │  ← Run in PARALLEL
│  Intent │ Urgency │ Sentiment       │
└────────┬────────────────────────────┘
         │ Classifications
         ▼
┌─────────────────┐
│   RAG Engine    │  ← ChromaDB + BM25 hybrid search over PMJAY PDFs
│   (ChromaDB)    │
└────────┬────────┘
         │ Grounded Answer
         ▼
┌─────────────────┐
│   SLA Engine    │  ← Auto-resolve OR generate AI action plan
│  (sla_engine)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI Backend│────▶│  React Dashboard │
│  + SQLite DB    │     │  + TTS readback  │
└─────────────────┘     └──────────────────┘
```

---

## 🚀 Features

### Core Pipeline
- **Whisper `tiny` + int8** — fast CPU transcription (~2-4s for short queries)
- **Parallel BERT inference** — all 3 classifiers run simultaneously via `ThreadPoolExecutor`
- **Hybrid RAG** — ChromaDB semantic search + BM25 keyword search over PMJAY PDFs
- **Web Speech API TTS** — full RAG answer read aloud to the agent

### SLA Engine (4-Level Decision Framework)
| Level | Description |
|-------|-------------|
| 1 | Intent + urgency override matrix (e.g. Escalation + Critical → 1hr SLA) |
| 2 | Sentiment bump — Negative sentiment escalates urgency one level |
| 3 | Smart auto-resolution for routine intents when RAG has a confident answer |
| 4 | Groq Llama 3.3 generates specific action plans for complex cases |

### Auto-Resolution
Routine intents resolved automatically — no human needed:
- `Coverage_Query`
- `Policy_Query`
- `Document_Query`
- `Claim_Status_Query`
- `Process_Clarification`
- `Positive_Feedback`

### AI Action Plans
Complex intents get a Groq-generated resolution plan:
- `Complaint`
- `Escalation_Request`
- `Reimbursement_Query`
- `Dispute_Clarification`

### Autonomous Background Tasks
- Duplicate call detection (same caller + intent within 24h)
- Repeat caller escalation (3+ unresolved → supervisor)
- Missed callback auto-rescheduling (every 30 min)
- Stale case auto-close (after 7 days resolved)

---

## 📁 Project Structure

```
Claim-Dashboard/
│
├── main.py                  # FastAPI backend — all API endpoints
├── pipeline.py              # Whisper + parallel DistilBERT + RAG pipeline
├── sla_engine.py            # SLA decision engine + Groq action plan generation
├── db.py                    # SQLite database layer
├── 2_RAG.py                 # RAG module (ChromaDB + BM25 hybrid search)
│
├── claimdesk_dashboard.html # Main agent dashboard (React, single-file)
├── case_detail.html         # Case detail + resolution page (opens in new tab)
│
├── claimdesk.db             # SQLite database (auto-created on first run)
├── .env                     # Environment variables (see below)
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Anaconda (recommended)
- A modern browser (Chrome/Edge for best TTS support)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/claimdesk-ai.git
cd claimdesk-ai
```

### 2. Create and activate environment
```bash
conda create -n claimdesk python=3.10 -y
conda activate claimdesk
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn python-dotenv faster-whisper transformers torch \
            chromadb sentence-transformers huggingface_hub rank_bm25 \
            numpy scipy pydantic
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# Model paths (required)
INTENT_MODEL_PATH=D:\models\intent
URGENCY_MODEL_PATH=D:\models\urgency
SENTIMENT_MODEL_PATH=D:\models\sentiment

# Groq API key (for AI action plan generation)
GROQ_API_KEY=gsk_your_groq_api_key_here

# SLA configuration (optional — defaults shown)
MAX_CALLS_PER_AGENT_PER_HOUR=5
REPEAT_CALL_HOURS=24
REPEAT_ESCALATE_N=3
STALE_CLOSE_DAYS=7
```

> **Get your Groq API key free at:** https://console.groq.com → API Keys → Create

### 5. Prepare BERT models

Place your fine-tuned DistilBERT models in the paths specified in `.env`. Each model directory should contain:
```
intent/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer_config.json
├── vocab.txt
└── label_map.json       ← {0: "Coverage_Query", 1: "Complaint", ...}
```

### 6. Prepare PMJAY knowledge base

Ensure `2_RAG.py` is in the project root alongside `main.py`. This file handles ChromaDB initialization and the `answer(query)` function.

---

## ▶️ Running the System

### Terminal 1 — Start the backend
```bash
cd Claim-Dashboard
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
Loading Whisper tiny (int8)...
Loading classifiers (parallel-ready)...
Loading RAG...
Found RAG file at: C:\...\Claim-Dashboard\2_RAG.py
Pipeline ready.
✅ DB ready: claimdesk.db
```

### Terminal 2 — Serve the dashboard
```bash
cd Claim-Dashboard
python -m http.server 3000
```

### Open in browser
```
http://localhost:3000/claimdesk_dashboard.html
```

---

## 🖥️ Dashboard Guide

### Main Dashboard
| Panel | Description |
|-------|-------------|
| 🎙 Voice Assistant | Click mic → speak query → click stop |
| 🧠 Analysis | Real-time intent / urgency / sentiment with confidence |
| 📚 Knowledge Base | Full RAG answer from PMJAY documents + sources |
| 📅 Follow-up Queue | Pending cases sorted by urgency |
| 📊 Analytics | Intent distribution, sentiment trend, resolution rate |
| 🗂 Case History | All cases — click to open detail page |

### Case Detail Page (`case_detail.html`)
Opens in a new tab when clicking any case or "Resolve" button. Shows:
- AI Analysis (intent / urgency / sentiment)
- Full transcript
- Knowledge base answer
- 🤖 AI Action Plan (for complex cases)
- Resolution checklist
- Agent response composer

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/demo/stream` | Submit audio → SSE stream of pipeline results |
| `GET` | `/api/analytics` | Dashboard metrics |
| `GET` | `/api/cases` | All cases |
| `GET` | `/api/cases/{ref}` | Single case |
| `GET` | `/api/cases/{ref}/calls` | Calls for a case |
| `GET` | `/api/cases/{ref}/followups` | Followups for a case |
| `GET` | `/api/followups` | All followups |
| `PATCH` | `/api/followups/{id}` | Resolve a followup |
| `POST` | `/api/maintenance/run` | Trigger autonomous maintenance |
| `GET` | `/api/maintenance/status` | Maintenance status |
| `GET` | `/health` | Backend health check |

---

## 🧠 ML Models

### Intent Classifier
Fine-tuned DistilBERT for 10-class classification:

| Intent | Description |
|--------|-------------|
| `Coverage_Query` | What is covered under PMJAY? |
| `Claim_Status_Query` | What is the status of my claim? |
| `Reimbursement_Query` | How do I get reimbursed? |
| `Document_Query` | What documents do I need? |
| `Complaint` | I want to file a complaint |
| `Escalation_Request` | I want to speak to a supervisor |
| `Policy_Query` | What are the policy terms? |
| `Process_Clarification` | How does the process work? |
| `Dispute_Clarification` | I want to dispute a decision |
| `Positive_Feedback` | Thank you / good experience |

### Urgency Classifier
4-class: `Critical` → `High` → `Medium` → `Low`

### Sentiment Classifier
3-class: `Positive` / `Neutral` / `Negative`

---

## 🔄 Complete Call Flow

```
1. Agent clicks mic → browser captures audio (WebM/Opus)
2. Audio POSTed to /api/demo/stream (SSE endpoint)
3. Whisper transcribes → SSE event: {step: "transcript"}
4. 3 DistilBERT models run in parallel → SSE event: {step: "classified"}
5. RAG retrieves PMJAY answer → SSE event: {step: "done", rag_answer: "..."}
6. SLA engine decides:
   ├── Routine intent + RAG answer → auto_resolved = True
   └── Complex intent → Groq generates action plan → stored in followup
7. TTS reads full RAG answer aloud
8. Dashboard updates: analysis panel, knowledge base, follow-up queue
9. For complex cases: agent clicks Resolve → case_detail.html opens
10. Agent reviews AI action plan → approves → case closed
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| End-to-end latency | ~6-10 seconds (CPU) |
| Whisper transcription | ~2-4s (tiny, int8) |
| Parallel BERT inference | ~0.5s (3 models simultaneously) |
| RAG retrieval | ~1-2s |
| Auto-resolution rate | ~60-70% of routine queries |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | OpenAI Whisper (faster-whisper, tiny, int8) |
| Intent Classification | DistilBERT (fine-tuned) |
| Urgency Classification | DistilBERT (fine-tuned) |
| Sentiment Analysis | DistilBERT (fine-tuned) |
| Vector Database | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| Action Plan LLM | Llama 3.3 70B via Groq API |
| Backend | FastAPI + SQLite |
| Frontend | React 18 (single HTML file, Babel) |
| Text-to-Speech | Web Speech API (browser native) |
| Real-time Updates | Server-Sent Events (SSE) + WebSocket |

---

## 🌐 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INTENT_MODEL_PATH` | ✅ | — | Path to fine-tuned intent model |
| `URGENCY_MODEL_PATH` | ✅ | — | Path to fine-tuned urgency model |
| `SENTIMENT_MODEL_PATH` | ✅ | — | Path to fine-tuned sentiment model |
| `GROQ_API_KEY` | ⚠️ | `""` | Groq API key (fallback plan used if empty) |
| `MAX_CALLS_PER_AGENT_PER_HOUR` | ❌ | `5` | Agent capacity per slot |
| `REPEAT_CALL_HOURS` | ❌ | `24` | Duplicate call detection window |
| `REPEAT_ESCALATE_N` | ❌ | `3` | Unresolved calls before escalation |
| `STALE_CLOSE_DAYS` | ❌ | `7` | Days before resolved cases auto-close |

---
## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [HuggingFace Transformers](https://huggingface.co/transformers/) — DistilBERT models
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Groq](https://groq.com/) — Fast LLM inference (Llama 3.3)
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
- National Health Authority — PMJAY policy documents

---

*Built for the PMJAY claims processing problem domain as part of NLP coursework, Semester 6.*
