import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pyttsx3
import joblib



# ==========================
# LOAD MODELS (LOAD ONCE)
# ==========================

whisper_model = WhisperModel("base", device="cpu")

intent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

intent_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_intent_major\checkpoint-300"
)

urgency_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

urgency_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_urgency\checkpoint-600"
)

sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_sentiment\checkpoint-600"
)

engine = pyttsx3.init()



intent_labels = {
    0: 'Claim_Status_Query',
    1: 'Complaint',
    2: 'Coverage_Query',
    3: 'Dispute_Clarification',
    4: 'Document_Query',
    5: 'Escalation_Request',
    6: 'Policy_Query',
    7: 'Positive_Feedback',
    8: 'Process_Clarification',
    9: 'Reimbursement_Query'
}


urgency_labels = {
    0: 'Critical',
    1: 'High',
    2: 'Low',
    3: 'Medium'
}

sentiment_labels = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}


emergency_keywords = ["icu", "emergency", "surgery", "critical", "life threatening"]

problem_keywords = ["not approved", "not received", "pending", "delay", "refused", "rejected"]
# ==========================
# RECORD AUDIO
# ==========================

def record_audio(filename="live_audio.wav", duration=5, fs=16000):
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Recording complete.")

# ==========================
# PREDICT FUNCTION
# ==========================

def predict(text, model, tokenizer,label_dict):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits,dim=1)
    confidence = torch.max(probabilities).item()
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if confidence <0.50:
        return "Uncertain"
    return label_dict[predicted_class]



# intent_label = predict(full_text, intent_model, intent_tokenizer,intent_labels)
# urgency_label = predict(full_text, urgency_model, urgency_tokenizer,urgency_labels)
# ==========================
# SPEAK FUNCTION
# ==========================

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ==========================
# MAIN LOOP
# ==========================

while True:
    record_audio()

    segments, info = whisper_model.transcribe("live_audio.wav")
    full_text = " ".join([segment.text for segment in segments])

    print("\n📝 You said:", full_text)

    if full_text.strip() == "":
        continue

    if "exit" in full_text.lower():
        speak("Shutting down assistant. Goodbye.")
        break
    
    if len(full_text.split()) < 3:
       speak("I could not understand clearly. Please repeat.")
       continue

    # Predict intent and urgency
    intent_label = predict(full_text, intent_model, intent_tokenizer,intent_labels)
    urgency_label = predict(full_text, urgency_model, urgency_tokenizer,urgency_labels)
    sentiment_label = predict(full_text, sentiment_model, sentiment_tokenizer,sentiment_labels)


    print("Intent:", intent_label)
    print("Urgency:", urgency_label)
    print("Sentiment:",sentiment_label)

    # SLA Logic (inline, no extra function)
    if urgency_label == "Critical":
        sla = "within 2 hours"
    elif urgency_label == "High":
        sla = "within 6 hours"
    elif urgency_label == "Medium":
        sla = "within 24 hours"
    else:
        sla = "within 48 hours"

    response = f"Your request is categorized as {intent_label}. It will be resolved {sla}."

    print("🤖 Assistant:", response)
    speak(response)