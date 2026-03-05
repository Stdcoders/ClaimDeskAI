import uuid
from datetime import datetime, timedelta
from config import SLA_RULES, FOLLOWUP_RULES
from database import create_case
from scheduler_engine import schedule_escalation, schedule_followup


def assign_agent(intent):

    routing_rules = {
        "Claim_Status_Query": "Claims Team",
        "Complaint": "Escalation Team",
        "Reimbursement_Query": "Finance Team",
        "Policy_Query": "Policy Support",
        "Document_Query": "Documentation Team"
    }

    return routing_rules.get(intent, "General Support")


def process_case(transcript, intent, urgency, sentiment, scheduler):

    # 1️⃣ Calculate SLA
    sla_hours = SLA_RULES.get(urgency, 48)
    due_time = datetime.now() + timedelta(hours=sla_hours)

    # 2️⃣ Assign agent
    assigned_agent = assign_agent(intent)

    # 3️⃣ Create ticket in DB
    case_id = str(uuid.uuid4())

    create_case(
        case_id=case_id,
        transcript=transcript,
        intent=intent,
        urgency=urgency,
        sentiment=sentiment,
        sla_hours=sla_hours,
        assigned_agent=assigned_agent,
        due_time=due_time
    )

    # 4️⃣ Schedule escalation
    schedule_escalation(
        scheduler,
        case_id,
        due_time
    )

    # 5️⃣ Schedule follow-up
    followup_hours = FOLLOWUP_RULES.get(urgency, 24)
    followup_time = datetime.now() + timedelta(hours=followup_hours)

    schedule_followup(
        scheduler,
        case_id,
        followup_time
    )

    return {
        "case_id": case_id,
        "assigned_agent": assigned_agent,
        "sla_hours": sla_hours,
        "due_time": due_time
    }
