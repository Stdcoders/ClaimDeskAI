import sqlite3
from datetime import datetime


DB_NAME = "claimdesk.db"


# ==========================
# INITIALIZE DATABASE
# ==========================

def init_db():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Cases Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            transcript TEXT,
            intent TEXT,
            urgency TEXT,
            sentiment TEXT,
            sla_hours INTEGER,
            status TEXT,
            assigned_agent TEXT,
            created_at TEXT,
            due_at TEXT
        )
    """)

    # Followups Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS followups (
            followup_id TEXT PRIMARY KEY,
            case_id TEXT,
            followup_time TEXT,
            status TEXT
        )
    """)

    # Escalations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            escalation_id TEXT PRIMARY KEY,
            case_id TEXT,
            escalation_time TEXT,
            status TEXT
        )
    """)

    conn.commit()
    conn.close()
def create_case(case_id, transcript, intent, urgency, sentiment,
                sla_hours, assigned_agent, due_time):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        transcript,
        intent,
        urgency,
        sentiment,
        sla_hours,
        "Open",
        assigned_agent,
        datetime.now().isoformat(),
        due_time.isoformat()
    ))

    conn.commit()
    conn.close()
def create_followup(followup_id, case_id, followup_time):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO followups VALUES (?, ?, ?, ?)
    """, (
        followup_id,
        case_id,
        followup_time.isoformat(),
        "Pending"
    ))

    conn.commit()
    conn.close()
def create_escalation(escalation_id, case_id, escalation_time):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?)
    """, (
        escalation_id,
        case_id,
        escalation_time.isoformat(),
        "Scheduled"
    ))

    conn.commit()
    conn.close()
