# sentiment_realtime.py
from transformers import pipeline
from collections import deque

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=-1
)

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

sentiment_history = deque(maxlen=5)

def analyze_sentiment(text):
    if not text or len(text.strip()) < 3:
        return None, 0.0

    result = sentiment_pipeline(text[:512])[0]
    return LABEL_MAP[result["label"]], round(result["score"], 3)

def update_and_check(text):
    sentiment, score = analyze_sentiment(text)
    if sentiment:
        sentiment_history.append(sentiment)

    escalate = sentiment_history.count("negative") >= 3
    return sentiment, score, escalate
