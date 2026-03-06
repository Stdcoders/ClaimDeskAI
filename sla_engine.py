"""
sla_engine.py — Advanced SLA: Level 3 (workload-aware) + Level 4 (auto-resolution)
====================================================================================
Level 3: Finds the earliest available agent slot under capacity threshold
Level 4: Auto-resolves routine intents without scheduling a human callback
"""

import os
from datetime import datetime, timedelta
from db import get_all_agents, get_agent_schedule, get_all_cases

MAX_PER_SLOT = int(os.getenv("MAX_CALLS_PER_AGENT_PER_HOUR", 5))

# ── Base SLA by urgency (hours) ───────────────────────────────────────────────
BASE_SLA_HOURS = {
    "Critical": 2,
    "High":     6,
    "Medium":   24,
    "Low":      48,
}

# ── Level 1 override matrix: intent + urgency → adjusted hours ────────────────
INTENT_URGENCY_OVERRIDE = {
    ("Escalation_Request", "Critical"): 1,
    ("Escalation_Request", "High"):     3,
    ("Complaint",          "Critical"): 1,
    ("Complaint",          "High"):     4,
    ("Reimbursement_Query","Critical"): 2,
    ("Reimbursement_Query","High"):     5,
}

# ── Level 4 auto-resolvable intents ──────────────────────────────────────────
AUTO_RESOLVE_INTENTS = {
    "Positive_Feedback": {
        "action":  "log_and_close",
        "message": None,   # no SMS needed
        "needs_callback": False,
    },
    "Policy_Query": {
        "action":  "rag_answer_sufficient",
        "message": "Your policy query has been answered. Please check your SMS for details.",
        "needs_callback": False,
    },
    "Document_Query": {
        "action":  "send_document_checklist",
        "message": (
            "ClaimDesk AI: For your PMJAY claim, please submit: "
            "1) Aadhaar card, 2) PMJAY e-card, 3) Discharge summary, "
            "4) Original bills & receipts, 5) Doctor prescription. "
            "Reply HELP for assistance."
        ),
        "needs_callback": False,
    },
    "Claim_Status_Query": {
        "action":  "send_status_update",
        "message": "ClaimDesk AI: Your claim status update has been sent to you.",
        "needs_callback": False,
    },
}

# Sentiment bump: if Negative, escalate urgency one level
URGENCY_ESCALATION = {
    "Low":      "Medium",
    "Medium":   "High",
    "High":     "Critical",
    "Critical": "Critical",
}


def compute_effective_urgency(intent: str, urgency: str, sentiment: str) -> str:
    """Level 1: bump urgency up one level if sentiment is Negative."""
    effective = urgency
    if sentiment == "Negative" and urgency != "Critical":
        effective = URGENCY_ESCALATION.get(urgency, urgency)
    return effective


def compute_sla_deadline(intent: str, urgency: str, sentiment: str) -> tuple[str, int]:
    """
    Returns (sla_deadline_str, hours).
    Uses intent+urgency override matrix first, then base SLA.
    """
    effective_urgency = compute_effective_urgency(intent, urgency, sentiment)
    hours = INTENT_URGENCY_OVERRIDE.get(
        (intent, effective_urgency),
        BASE_SLA_HOURS.get(effective_urgency, 48)
    )
    deadline = datetime.now() + timedelta(hours=hours)
    return deadline.strftime("%Y-%m-%d %H:%M"), hours


def find_available_slot(agent_id: int, deadline: datetime) -> str | None:
    """
    Level 3: Walk hour slots from now until deadline.
    Return the first slot where agent has < MAX_PER_SLOT callbacks.
    Returns None if no slot available (→ escalate to supervisor).
    """
    schedule = get_agent_schedule(agent_id)
    now      = datetime.now().replace(minute=0, second=0, microsecond=0)
    current  = now + timedelta(hours=1)  # start from next hour

    while current <= deadline:
        slot_key = current.strftime("%Y-%m-%d %H")
        count    = schedule.get(slot_key, 0)
        if count < MAX_PER_SLOT:
            # Schedule at start of slot + offset by existing count (spread within hour)
            offset   = count * (60 // MAX_PER_SLOT)
            slot_dt  = current + timedelta(minutes=offset)
            return slot_dt.strftime("%Y-%m-%d %H:%M")
        current += timedelta(hours=1)

    return None  # all slots full


def assign_best_agent(deadline: datetime) -> tuple[int | None, str | None]:
    """
    Level 3: Find the least-loaded active agent who has a slot before deadline.
    Returns (agent_id, callback_dt) or (None, None) if all full → supervisor.
    """
    agents = get_all_agents()
    best_agent_id = None
    best_slot     = None
    best_load     = float("inf")

    for agent in agents:
        if agent["name"] == "Supervisor":
            continue
        schedule  = get_agent_schedule(agent["id"])
        total_load = sum(schedule.values())
        slot      = find_available_slot(agent["id"], deadline)
        if slot and total_load < best_load:
            best_load     = total_load
            best_agent_id = agent["id"]
            best_slot     = slot

    if not best_agent_id:
        # All agents full — escalate to supervisor
        supervisors = [a for a in agents if a["name"] == "Supervisor"]
        if supervisors:
            sup_slot = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
            return supervisors[0]["id"], sup_slot

    return best_agent_id, best_slot


def process_call_outcome(
    call_id: int,
    caller_number: str,
    case_ref: str,
    intent: str,
    urgency: str,
    sentiment: str,
    transcript: str,
    rag_answer: str,
    twilio_client=None,
) -> dict:
    """
    Main SLA engine entry point. Called after every call ends.

    Returns a dict describing what action was taken:
    {
        auto_resolved: bool,
        auto_action: str,
        needs_callback: bool,
        agent_id: int | None,
        callback_dt: str | None,
        sla_deadline: str,
        sms_sent: bool,
        message: str,
    }
    """
    result = {
        "auto_resolved":  False,
        "auto_action":    None,
        "needs_callback": True,
        "agent_id":       None,
        "callback_dt":    None,
        "sla_deadline":   None,
        "sms_sent":       False,
        "message":        "",
    }

    # ── Level 4: Check for auto-resolvable intent ─────────────────────────────
    if intent in AUTO_RESOLVE_INTENTS:
        rule = AUTO_RESOLVE_INTENTS[intent]
        result["auto_resolved"]  = True
        result["auto_action"]    = rule["action"]
        result["needs_callback"] = rule["needs_callback"]
        result["message"]        = f"Auto-resolved: {rule['action']}"

        # Send SMS if applicable
        if rule["message"] and twilio_client and caller_number:
            try:
                sms_body = rule["message"]
                # For Claim_Status_Query — inject actual case status
                if intent == "Claim_Status_Query":
                    cases = get_all_cases()
                    matching = [c for c in cases if c["caller_number"] == caller_number]
                    if matching:
                        c = matching[0]
                        sms_body = (
                            f"ClaimDesk AI: Your case {c['case_ref']} is currently "
                            f"'{c['status']}' (Priority: {c['priority']}). "
                            f"Last updated: {c['updated_at'][:16]}."
                        )
                twilio_client.messages.create(
                    body=sms_body,
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    to=caller_number
                )
                result["sms_sent"] = True
            except Exception as e:
                print(f"SMS error: {e}")

        # Compute SLA for record-keeping even if auto-resolved
        sla_deadline, _ = compute_sla_deadline(intent, urgency, sentiment)
        result["sla_deadline"] = sla_deadline
        return result

    # ── Level 3: Workload-aware scheduling ───────────────────────────────────
    effective_urgency      = compute_effective_urgency(intent, urgency, sentiment)
    sla_deadline_str, hours = compute_sla_deadline(intent, urgency, sentiment)
    sla_deadline_dt         = datetime.now() + timedelta(hours=hours)

    result["sla_deadline"] = sla_deadline_str

    # Handle Critical + Complaint → auto-SMS supervisor alert
    if effective_urgency == "Critical" and intent in ("Complaint", "Escalation_Request"):
        if twilio_client:
            try:
                twilio_client.messages.create(
                    body=(
                        f"🚨 URGENT ClaimDesk Alert: {intent} call from {caller_number}. "
                        f"Case: {case_ref}. Sentiment: {sentiment}. "
                        f"Transcript: {transcript[:200]}"
                    ),
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    to=os.getenv("SUPERVISOR_PHONE")
                )
                result["sms_sent"]    = True
                result["auto_action"] = "supervisor_alerted"
            except Exception as e:
                print(f"Supervisor SMS error: {e}")

    # Find best available agent slot
    agent_id, callback_dt = assign_best_agent(sla_deadline_dt)
    result["agent_id"]    = agent_id
    result["callback_dt"] = callback_dt

    if callback_dt:
        result["message"] = f"Scheduled with agent {agent_id} at {callback_dt}"
    else:
        result["message"] = "All slots full — escalated to supervisor"
        result["auto_action"] = "escalated_to_supervisor"

    return result