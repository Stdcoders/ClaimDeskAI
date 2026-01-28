from RealtimeSTT import AudioToTextRecorder
import csv
from datetime import datetime
import uuid
import os
from sentiment1 import update_and_check

CSV_FILE = "call_transcripts.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "call_id",
            "audio_source",
            "channel",
            "transcript_raw",
            "timestamp"
        ])

def on_update(text):
    print(f"\rTranscribing: {text}", end="", flush=True)
    sentiment, score, escalate = update_and_check(text)

    if sentiment:
        print(f"\n📊 Sentiment: {sentiment} ({score})")

    if escalate:
        print("🚨 Supervisor alert triggered")
def on_finalized(text):
    
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Make transcript CSV-safe
    transcript = text.replace("\n", " ").strip()

    # Save to CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            call_id,
            "mic",
            "call",
            transcript,
            timestamp
        ])

    print(f"\n✅ Final saved ({call_id}): {transcript}")

if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_update,
        spinner=False
    )

    print("🎙️ Listening... Press Ctrl+C to stop.\n")

    while True:
        recorder.text(on_finalized)
