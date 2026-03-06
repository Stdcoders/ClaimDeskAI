import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import json
import torch
import pyttsx3
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================
# LOAD MODELS (LOAD ONCE)
# ==========================

print("Loading Whisper...")
whisper_model = WhisperModel("base", device="cpu")

print("Loading intent model...")
intent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
intent_model     = AutoModelForSequenceClassification.from_pretrained(r"D:\models\intent")

print("Loading urgency model...")
urgency_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
urgency_model     = AutoModelForSequenceClassification.from_pretrained(r"D:\models\urgency")

print("Loading sentiment model...")
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentiment_model     = AutoModelForSequenceClassification.from_pretrained(r"D:\models\sentiment")

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate',   150)   # slowed from default ~200
engine.setProperty('volume', 1.0)

# Load label maps from trained model folders — always in sync with training
with open(r"D:\models\intent\label_map.json") as f:
    intent_labels = {int(k): v for k, v in json.load(f).items()}

with open(r"D:\models\urgency\label_map.json") as f:
    urgency_labels = {int(k): v for k, v in json.load(f).items()}

with open(r"D:\models\sentiment\label_map.json") as f:
    sentiment_labels = {int(k): v for k, v in json.load(f).items()}

print("\n✅ All models loaded.\n")

# Keywords for override logic
emergency_keywords = ["icu", "emergency", "surgery", "critical", "life threatening"]
problem_keywords   = ["not approved", "not received", "pending", "delay", "refused", "rejected"]


# ==========================
# RECORD AUDIO
# ==========================

def record_audio(filename="live_audio.wav", duration=8, fs=16000):
    print("─" * 40)
    print("🎤  Speak now... (recording for 8 seconds)")
    time.sleep(0.5)   # small pause so print output doesn't get picked up
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    # Convert float32 → int16 for wav
    wav.write(filename, fs, (recording * 32767).astype(np.int16))
    print("⏹  Recording complete.")


# ==========================
# PREDICT FUNCTION
# ==========================

def predict(text, model, tokenizer, label_dict, threshold=0.30):
    """
    Run classifier and return (label, confidence).
    Returns ("Uncertain", confidence) if below threshold.
    """
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


# ==========================
# SPEAK FUNCTION
# ==========================

def speak(text):
    engine.say(text)
    engine.runAndWait()


# ==========================
# SLA HELPER
# ==========================

def get_sla(urgency_label):
    return {
        "Critical": "within 2 hours",
        "High":     "within 6 hours",
        "Medium":   "within 24 hours",
        "Low":      "within 48 hours",
    }.get(urgency_label, "within 48 hours")


# ==========================
# MAIN LOOP
# ==========================

print("🏥 ClaimDesk AI — Voice Assistant")
print("Say 'exit' to quit.\n")

while True:

    record_audio()

    # Transcribe
    segments, _ = whisper_model.transcribe("live_audio.wav", language="en")
    full_text   = " ".join([s.text for s in segments]).strip()

    print(f"\n📝 You said: {full_text}")

    # Empty audio
    if not full_text:
        speak("I did not hear anything. Please try again.")
        continue

    # Exit command
    if "exit" in full_text.lower():
        speak("Shutting down assistant. Goodbye.")
        break

    # Too short to classify reliably
    if len(full_text.split()) < 3:
        speak("I could not understand clearly. Please speak a full sentence.")
        continue

    # Classify
    intent_label,    ci = predict(full_text, intent_model,    intent_tokenizer,    intent_labels)
    urgency_label,   cu = predict(full_text, urgency_model,   urgency_tokenizer,   urgency_labels)
    sentiment_label, cs = predict(full_text, sentiment_model, sentiment_tokenizer, sentiment_labels)

    # Keyword overrides — catch obvious cases the model might miss
    text_lower = full_text.lower()
    if any(kw in text_lower for kw in emergency_keywords) and urgency_label not in ("Critical", "High"):
        urgency_label = "High"
        cu = 1.0
        print("⚡ Urgency overridden to High (emergency keyword detected)")

    # Print results with confidence
    print(f"\n  Intent   : {intent_label:<28} ({ci:.0%})")
    print(f"  Urgency  : {urgency_label:<28} ({cu:.0%})")
    print(f"  Sentiment: {sentiment_label:<28} ({cs:.0%})")

    # Build response
    sla = get_sla(urgency_label)

    if intent_label == "Uncertain":
        response = f"I have noted your concern. It will be reviewed {sla}."
    else:
        intent_readable = intent_label.replace("_", " ")
        response = f"Your {intent_readable} has been noted and will be resolved {sla}."

    # Add sentiment-aware tone
    if sentiment_label == "Negative":
        response = "I understand your frustration. " + response
    elif sentiment_label == "Positive":
        response = "Thank you for reaching out. " + response

    print(f"\n🤖 Assistant: {response}")
    speak(response)