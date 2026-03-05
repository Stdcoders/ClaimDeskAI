# SLA rules in hours
SLA_RULES = {
    "Critical": 2,
    "High": 6,
    "Medium": 24,
    "Low": 48,
    "Uncertain": 48
}

# Follow-up rules
FOLLOWUP_RULES = {
    "Critical": 1,    # Follow up in 1 hour
    "High": 3,
    "Medium": 12,
    "Low": 24
}

# Escalation triggers
ESCALATION_CONDITIONS = {
    "negative_sentiment": True,
    "critical_intent": ["Escalation_Request", "Complaint"]
}
