"""
twilio_handler.py — Inbound & Outbound call handling
=====================================================
Inbound:  Twilio calls /call/inbound webhook → records → transcribes → pipeline
Outbound: POST /call/outbound → Twilio REST API initiates call to customer
"""

import os
import uuid
import asyncio
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather, Record

ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER")
BACKEND_URL    = os.getenv("BACKEND_URL", "http://localhost:8000")

_client = None

def get_twilio_client() -> Client:
    global _client
    if _client is None:
        _client = Client(ACCOUNT_SID, AUTH_TOKEN)
    return _client


# ── Inbound call TwiML ────────────────────────────────────────────────────────

def build_inbound_twiml(call_id: int) -> str:
    """
    TwiML response for inbound calls:
    1. Greet the caller
    2. Record their message (transcription via Whisper on our side)
    3. Redirect to /call/recording_complete when done
    """
    resp   = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"{BACKEND_URL}/call/speech_result/{call_id}",
        method="POST",
        speech_timeout="auto",
        language="en-IN",
        enhanced=True,
    )
    gather.say(
        "Welcome to ClaimDesk AI. Please describe your claim query after the tone, "
        "and press any key when finished.",
        voice="Polly.Aditi",
        language="en-IN"
    )
    resp.append(gather)

    # Fallback if no input
    resp.say("We did not receive your input. Please call back and try again.")
    return str(resp)


def build_recording_twiml(call_id: int) -> str:
    """TwiML for recording full call audio — used as fallback."""
    resp = VoiceResponse()
    resp.say("Please describe your query after the beep.", voice="Polly.Aditi")
    resp.record(
        action=f"{BACKEND_URL}/call/recording_complete/{call_id}",
        method="POST",
        max_length=120,
        play_beep=True,
        transcribe=False,   # we do our own transcription with Whisper
    )
    return str(resp)


def build_response_twiml(response_text: str) -> str:
    """TwiML to speak the RAG answer back to caller."""
    resp = VoiceResponse()
    resp.say(response_text, voice="Polly.Aditi", language="en-IN")
    resp.pause(length=1)
    resp.say(
        "Is there anything else I can help you with? Press 1 to ask another question, "
        "or hang up to end the call.",
        voice="Polly.Aditi"
    )
    gather = Gather(
        num_digits=1,
        action=f"{BACKEND_URL}/call/continue",
        method="POST",
        timeout=5,
    )
    resp.append(gather)
    resp.say("Thank you for calling ClaimDesk. Goodbye.", voice="Polly.Aditi")
    resp.hangup()
    return str(resp)


# ── Outbound call ─────────────────────────────────────────────────────────────

def make_outbound_call(to_number: str, call_id: int, message: str = None) -> str:
    """
    Initiate an outbound callback call via Twilio REST API.
    Returns the Twilio call SID.
    """
    client = get_twilio_client()

    if not message:
        message = (
            "Hello, this is ClaimDesk AI calling regarding your recent claim query. "
            "An agent will be with you shortly. Please hold."
        )

    # TwiML for outbound
    resp = VoiceResponse()
    resp.say(message, voice="Polly.Aditi", language="en-IN")
    resp.pause(length=2)

    gather = Gather(
        input="speech",
        action=f"{BACKEND_URL}/call/speech_result/{call_id}",
        method="POST",
        speech_timeout="auto",
        language="en-IN",
    )
    gather.say(
        "Please describe your query now, and press any key when finished.",
        voice="Polly.Aditi"
    )
    resp.append(gather)
    resp.say("We did not receive your response. We will call you again shortly.", voice="Polly.Aditi")

    twiml_xml = str(resp)

    call = client.calls.create(
        twiml=twiml_xml,
        to=to_number,
        from_=TWILIO_NUMBER,
        status_callback=f"{BACKEND_URL}/call/status/{call_id}",
        status_callback_method="POST",
    )
    return call.sid


def send_sms(to_number: str, message: str) -> bool:
    """Send SMS via Twilio."""
    try:
        client = get_twilio_client()
        client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=to_number
        )
        return True
    except Exception as e:
        print(f"SMS error: {e}")
        return False


def end_call(call_sid: str) -> bool:
    """Programmatically end an active call."""
    try:
        client = get_twilio_client()
        client.calls(call_sid).update(status="completed")
        return True
    except Exception as e:
        print(f"End call error: {e}")
        return False