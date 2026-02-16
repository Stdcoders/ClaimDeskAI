# nlp_engine.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================
# LOAD MODELS (LOAD ONCE)
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intent Model
intent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
intent_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_intent_major\checkpoint-300"
)
intent_model.to(DEVICE)
intent_model.eval()

# Urgency Model
urgency_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
urgency_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_urgency\checkpoint-600"
)
urgency_model.to(DEVICE)
urgency_model.eval()

# Sentiment Model
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\DELL\vs_code\AutoClaim\results_sentiment\checkpoint-600"
)
sentiment_model.to(DEVICE)
sentiment_model.eval()


# ==========================
# LABEL DICTIONARIES
# ==========================

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


# ==========================
# GENERIC PREDICT FUNCTION
# ==========================

def predict(text, model, tokenizer, label_dict):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probabilities).item()
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Confidence threshold
    if confidence < 0.50:
        return "Uncertain"

    return label_dict[predicted_class]


# ==========================
# MAIN NLP ENTRY FUNCTION
# ==========================

def analyze_text(text):
    """
    Takes transcript text and returns:
    (intent, urgency, sentiment)
    """

    intent = predict(text, intent_model, intent_tokenizer, intent_labels)
    urgency = predict(text, urgency_model, urgency_tokenizer, urgency_labels)
    sentiment = predict(text, sentiment_model, sentiment_tokenizer, sentiment_labels)

    return intent, urgency, sentiment
