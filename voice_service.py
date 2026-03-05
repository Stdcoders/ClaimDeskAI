import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import pyttsx3

# ==========================
# LOAD MODELS (ONLY ONCE)
# ==========================

whisper_model = WhisperModel("base", device="cpu")

tts_engine = pyttsx3.init()


# ==========================
# RECORD AUDIO
# ==========================

def record_audio(filename="live_audio.wav", duration=5, fs=16000):
    """
    Records audio from microphone and saves to WAV file.
    """
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("✅ Recording complete.")


# ==========================
# TRANSCRIBE AUDIO
# ==========================

def transcribe_audio(filename="live_audio.wav"):
    """
    Converts recorded audio to text using Whisper.
    """
    segments, info = whisper_model.transcribe(filename)

    full_text = " ".join([segment.text for segment in segments])
    full_text = full_text.strip()

    return full_text


# ==========================
# COMBINED FUNCTION
# ==========================

def record_and_transcribe():
    """
    Records audio and returns transcript.
    Handles basic validation.
    """

    record_audio()

    transcript = transcribe_audio()

    if not transcript:
        print("⚠️ No speech detected.")
        return None

    if len(transcript.split()) < 3:
        print("⚠️ Speech too short.")
        return None

    return transcript


# ==========================
# TEXT TO SPEECH
# ==========================

def speak(text):
    """
    Converts text to speech.
    """
    tts_engine.say(text)
    tts_engine.runAndWait()
