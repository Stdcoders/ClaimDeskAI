# main.py

from voice_service import record_and_transcribe, speak
from nlp_engine import analyze_text
from case_engine import process_case
from database import init_db
from scheduler_engine import start_scheduler


def main():

    # 1️⃣ Initialize system
    init_db()
    scheduler = start_scheduler()

    print("🚀 ClaimDesk-AI Voice Assistant Started")

    while True:

        # 2️⃣ Capture voice and convert to text
        transcript = record_and_transcribe()

        if not transcript:
            continue

        if "exit" in transcript.lower():
            speak("Shutting down Claim Desk. Goodbye.")
            break

        print("📝 Transcript:", transcript)

        # 3️⃣ Run NLP analysis
        intent, urgency, sentiment = analyze_text(transcript)

        print("Intent:", intent)
        print("Urgency:", urgency)
        print("Sentiment:", sentiment)

        # 4️⃣ Process case (SLA + DB + Assignment)
        case_data = process_case(
            transcript=transcript,
            intent=intent,
            urgency=urgency,
            sentiment=sentiment,
            scheduler=scheduler
        )

        # 5️⃣ Respond to user
        response = (
            f"Your case ID is {case_data['case_id'][:8]}. "
            f"It has been assigned to {case_data['assigned_agent']} "
            f"and will be resolved within {case_data['sla_hours']} hours."
        )

        print("🤖 Assistant:", response)
        speak(response)


if __name__ == "__main__":
    main()
