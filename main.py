"""
main.py — FastAPI backend for ClaimDesk AI
==========================================
INSTALL:
  pip install fastapi uvicorn python-dotenv twilio faster-whisper
              transformers torch chromadb sentence-transformers
              huggingface_hub rank_bm25

RUN:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import db
import pipeline as pipe
import twilio_handler as twilio
from sla_engine import process_call_outcome

# ── Active WebSocket connections: call_id → [ws, ws, ...] ────────────────────
active_connections: dict[int, list[WebSocket]] = {}


async def broadcast(call_id: int, data: dict):
    """Push update to all dashboard clients watching this call."""
    conns = active_connections.get(call_id, [])
    dead  = []
    for ws in conns:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for d in dead:
        conns.remove(d)


async def broadcast_all(data: dict):
    """Push update to ALL connected dashboard clients."""
    all_conns = [ws for conns in active_connections.values() for ws in conns]
    # Also broadcast to channel 0 (general dashboard)
    all_conns += active_connections.get(0, [])
    dead = []
    for ws in set(all_conns):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)


# ── App startup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    pipe.load_all()
    yield

app = FastAPI(title="ClaimDesk AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — live dashboard updates
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: int):
    await websocket.accept()
    if call_id not in active_connections:
        active_connections[call_id] = []
    active_connections[call_id].append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        active_connections[call_id].remove(websocket)


# ══════════════════════════════════════════════════════════════════════════════
# TWILIO WEBHOOKS — inbound calls
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/call/inbound")
async def inbound_call(request: Request):
    """
    Twilio calls this when a customer calls our number.
    Creates a call record, returns TwiML to greet and gather speech.
    """
    form = await request.form()
    call_sid      = form.get("CallSid", str(uuid.uuid4()))
    caller_number = form.get("From", "unknown")
    called_number = form.get("To", os.getenv("TWILIO_PHONE_NUMBER", ""))

    # Create DB record
    call_id = db.create_call(call_sid, "inbound", caller_number, called_number)

    # Get/create case for this caller
    case, is_new = db.get_or_create_case(caller_number)

    # Notify dashboard
    await broadcast_all({
        "event":         "new_call",
        "call_id":       call_id,
        "direction":     "inbound",
        "caller_number": caller_number,
        "case_ref":      case["case_ref"],
        "started_at":    datetime.now().isoformat(),
    })

    twiml = twilio.build_inbound_twiml(call_id)
    return Response(content=twiml, media_type="application/xml")


@app.post("/call/speech_result/{call_id}")
async def speech_result(call_id: int, request: Request):
    """
    Twilio posts transcribed speech here after caller speaks.
    We run our own pipeline: classify + RAG + SLA.
    """
    form          = await request.form()
    speech_result = form.get("SpeechResult", "")
    call_sid      = form.get("CallSid", "")

    if not speech_result:
        resp = twilio.build_recording_twiml(call_id)
        return Response(content=resp, media_type="application/xml")

    call = db.get_call(call_id)
    if not call:
        raise HTTPException(404, "Call not found")

    # Run pipeline on speech text (skip Whisper — Twilio already transcribed)
    classifications = pipe.classify_text(speech_result)
    rag_text, sources = pipe.get_rag_answer(speech_result, classifications["intent"])

    # Update call record
    db.update_call(call_id,
        transcript    = speech_result,
        intent        = classifications["intent"],
        urgency       = classifications["urgency"],
        sentiment     = classifications["sentiment"],
        conf_intent   = classifications["conf_intent"],
        conf_urgency  = classifications["conf_urgency"],
        conf_sentiment= classifications["conf_sentiment"],
        rag_context   = rag_text,
    )

    # Push live update to dashboard
    await broadcast(call_id, {
        "event":         "call_update",
        "call_id":       call_id,
        "transcript":    speech_result,
        "rag_answer":    rag_text,
        "sources":       sources,
        **classifications,
    })

    # Run SLA engine
    case, _ = db.get_or_create_case(call["caller_number"])
    sla_result = process_call_outcome(
        call_id        = call_id,
        caller_number  = call["caller_number"],
        case_ref       = case["case_ref"],
        intent         = classifications["intent"],
        urgency        = classifications["urgency"],
        sentiment      = classifications["sentiment"],
        transcript     = speech_result,
        rag_answer     = rag_text,
        twilio_client  = twilio.get_twilio_client(),
    )

    # Create follow-up record
    db.create_followup(
        case_ref           = case["case_ref"],
        caller_number      = call["caller_number"],
        call_id            = call_id,
        intent             = classifications["intent"],
        urgency            = classifications["urgency"],
        sentiment          = classifications["sentiment"],
        sla_deadline       = sla_result["sla_deadline"],
        callback_dt        = sla_result.get("callback_dt"),
        transcript_summary = speech_result[:500],
        agent_id           = sla_result.get("agent_id"),
        auto_resolved      = sla_result["auto_resolved"],
        auto_action        = sla_result.get("auto_action"),
    )

    # Update case
    db.update_case(case["case_ref"],
        last_intent   = classifications["intent"],
        last_sentiment= classifications["sentiment"],
        total_calls   = case["total_calls"] + 1,
    )

    # Notify dashboard of follow-up
    await broadcast_all({
        "event":        "followup_created",
        "case_ref":     case["case_ref"],
        "auto_resolved":sla_result["auto_resolved"],
        "callback_dt":  sla_result.get("callback_dt"),
        "sla_deadline": sla_result["sla_deadline"],
        "message":      sla_result["message"],
    })

    # Respond to caller with RAG answer
    response_twiml = twilio.build_response_twiml(rag_text[:500])
    return Response(content=response_twiml, media_type="application/xml")


@app.post("/call/status/{call_id}")
async def call_status(call_id: int, request: Request):
    """Twilio status callback — updates call status in DB."""
    form   = await request.form()
    status = form.get("CallStatus", "")
    duration = int(form.get("CallDuration", 0))

    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status in ("completed", "failed", "no-answer") else None
    db.update_call(call_id, status=status, duration_sec=duration, ended_at=ended_at)

    await broadcast_all({"event": "call_status", "call_id": call_id, "status": status, "duration": duration})
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# REST API — Dashboard data
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/calls")
def get_calls(limit: int = 50):
    return db.get_recent_calls(limit)

@app.get("/api/calls/{call_id}")
def get_call(call_id: int):
    call = db.get_call(call_id)
    if not call:
        raise HTTPException(404, "Call not found")
    return call

@app.get("/api/cases")
def get_cases(status: Optional[str] = None):
    return db.get_all_cases(status)

@app.get("/api/followups")
def get_followups(status: Optional[str] = None):
    return db.get_followups(status)

@app.get("/api/followups/overdue")
def get_overdue():
    return db.get_overdue_followups()

@app.patch("/api/followups/{followup_id}")
def update_followup(followup_id: int, body: dict):
    status = body.get("status")
    notes  = body.get("notes", "")
    if not status:
        raise HTTPException(400, "status required")
    db.update_followup_status(followup_id, status, notes)
    return {"ok": True}

@app.get("/api/analytics")
def get_analytics():
    return db.get_analytics()

@app.get("/api/agents")
def get_agents():
    return db.get_all_agents()


# ── Outbound call ─────────────────────────────────────────────────────────────

class OutboundCallRequest(BaseModel):
    to_number:  str
    message:    Optional[str] = None
    case_ref:   Optional[str] = None

@app.post("/api/call/outbound")
async def make_outbound_call(body: OutboundCallRequest):
    call_id = db.create_call(
        call_sid      = str(uuid.uuid4()),
        direction     = "outbound",
        caller_number = body.to_number,
        called_number = os.getenv("TWILIO_PHONE_NUMBER", ""),
    )
    try:
        call_sid = twilio.make_outbound_call(body.to_number, call_id, body.message)
        db.update_call(call_id, call_sid=call_sid)
        await broadcast_all({
            "event":     "new_call",
            "call_id":   call_id,
            "direction": "outbound",
            "to_number": body.to_number,
            "case_ref":  body.case_ref,
        })
        return {"call_id": call_id, "call_sid": call_sid}
    except Exception as e:
        raise HTTPException(500, f"Twilio error: {e}")


# ── Demo mode endpoint — browser mic audio ────────────────────────────────────

@app.post("/api/demo")
async def demo_endpoint(audio: UploadFile):
    """
    Receives audio blob from browser mic.
    Runs full pipeline: Whisper → BERT → RAG → SLA.
    Returns transcript + classifications + RAG answer.
    """
    import tempfile, shutil
    suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    result = pipe.run_pipeline(tmp_path)
    os.unlink(tmp_path)

    if "error" in result:
        return {"error": result["error"]}

    # Create demo call + case record
    call_id = db.create_call(
        call_sid      = f"demo-{uuid.uuid4()}",
        direction     = "demo",
        caller_number = "demo-user",
        called_number = "demo",
    )
    db.update_call(call_id,
        transcript     = result["transcript"],
        intent         = result["intent"],
        urgency        = result["urgency"],
        sentiment      = result["sentiment"],
        conf_intent    = result["conf_intent"],
        conf_urgency   = result["conf_urgency"],
        conf_sentiment = result["conf_sentiment"],
        rag_context    = result["rag_answer"],
        status         = "completed",
    )
    case, _ = db.get_or_create_case("demo-user")

    # Run SLA engine
    sla = process_call_outcome(
        call_id       = call_id,
        caller_number = "demo-user",
        case_ref      = case["case_ref"],
        intent        = result["intent"],
        urgency       = result["urgency"],
        sentiment     = result["sentiment"],
        transcript    = result["transcript"],
        rag_answer    = result["rag_answer"],
        twilio_client = None,
    )

    db.create_followup(
        case_ref           = case["case_ref"],
        caller_number      = "demo-user",
        call_id            = call_id,
        intent             = result["intent"],
        urgency            = result["urgency"],
        sentiment          = result["sentiment"],
        sla_deadline       = sla["sla_deadline"],
        callback_dt        = sla.get("callback_dt"),
        transcript_summary = result["transcript"][:300],
        auto_resolved      = sla["auto_resolved"],
        auto_action        = sla.get("auto_action"),
    )

    db.update_case(case["case_ref"],
        last_intent    = result["intent"],
        last_sentiment = result["sentiment"],
        total_calls    = case["total_calls"] + 1,
    )

    return {
        **result,
        "call_id":    call_id,
        "case_ref":   case["case_ref"],
        "sla":        sla,
    }



@app.get("/health")
def health():
    return {
        "status": "ok",
        "time":   datetime.now().isoformat(),
        "db":     os.path.exists(db.DB_PATH),
    }
