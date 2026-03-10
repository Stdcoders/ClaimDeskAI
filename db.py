"""
db.py — SQLite database for ClaimDesk AI
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "claimdesk.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            phone      TEXT,
            email      TEXT,
            is_active  INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid       TEXT UNIQUE,
            direction      TEXT,
            caller_number  TEXT,
            called_number  TEXT,
            agent_id       INTEGER,
            status         TEXT DEFAULT 'active',
            transcript     TEXT DEFAULT '',
            intent         TEXT,
            urgency        TEXT,
            sentiment      TEXT,
            conf_intent    REAL,
            conf_urgency   REAL,
            conf_sentiment REAL,
            rag_context    TEXT,
            duration_sec   INTEGER DEFAULT 0,
            started_at     TEXT DEFAULT (datetime('now','localtime')),
            ended_at       TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            case_ref      TEXT UNIQUE,
            caller_number TEXT,
            claim_id      TEXT,
            status        TEXT DEFAULT 'Open',
            priority      TEXT DEFAULT 'Medium',
            total_calls   INTEGER DEFAULT 0,
            last_intent   TEXT,
            last_sentiment TEXT,
            notes         TEXT DEFAULT '',
            created_at    TEXT DEFAULT (datetime('now','localtime')),
            updated_at    TEXT DEFAULT (datetime('now','localtime'))
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS followups (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            case_ref           TEXT,
            caller_number      TEXT,
            call_id            INTEGER,
            agent_id           INTEGER,
            intent             TEXT,
            urgency            TEXT,
            sentiment          TEXT,
            sla_deadline       TEXT,
            callback_dt        TEXT,
            transcript_summary TEXT,
            auto_resolved      INTEGER DEFAULT 0,
            auto_action        TEXT,
            status             TEXT DEFAULT 'Pending',
            resolution_notes   TEXT DEFAULT '',
            created_at         TEXT DEFAULT (datetime('now','localtime')),
            resolved_at        TEXT,
            FOREIGN KEY (call_id)  REFERENCES calls(id),
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        )""")

        count = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        if count == 0:
            conn.executemany("INSERT INTO agents (name,phone,email) VALUES (?,?,?)", [
                ("Agent Priya",  "+910000000001", "priya@claimdesk.ai"),
                ("Agent Rahul",  "+910000000002", "rahul@claimdesk.ai"),
                ("Agent Sneha",  "+910000000003", "sneha@claimdesk.ai"),
                ("Supervisor",   "+910000000004", "supervisor@claimdesk.ai"),
            ])
        conn.commit()
    print(f"✅ DB ready: {Path(DB_PATH).resolve()}")

def create_call(call_sid, direction, caller_number, called_number, agent_id=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calls (call_sid,direction,caller_number,called_number,agent_id) VALUES (?,?,?,?,?)",
            (call_sid, direction, caller_number, called_number, agent_id))
        conn.commit()
        return cur.lastrowid

def update_call(call_id, **kwargs):
    if not kwargs: return
    cols = ", ".join(f"{k}=?" for k in kwargs)
    with get_conn() as conn:
        conn.execute(f"UPDATE calls SET {cols} WHERE id=?", list(kwargs.values()) + [call_id])
        conn.commit()

def get_call(call_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    return dict(row) if row else None

def get_recent_calls(limit=50):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT c.*, a.name as agent_name FROM calls c
            LEFT JOIN agents a ON c.agent_id=a.id
            ORDER BY c.started_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]

def get_or_create_case(caller_number):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE caller_number=? ORDER BY created_at DESC LIMIT 1",
            (caller_number,)).fetchone()
        if row:
            return dict(row), False
        count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        case_ref = f"CASE-{datetime.now().year}-{count+1:04d}"
        conn.execute("INSERT INTO cases (case_ref,caller_number) VALUES (?,?)", (case_ref, caller_number))
        conn.commit()
        row = conn.execute("SELECT * FROM cases WHERE case_ref=?", (case_ref,)).fetchone()
        return dict(row), True

def update_case(case_ref, **kwargs):
    kwargs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ", ".join(f"{k}=?" for k in kwargs)
    with get_conn() as conn:
        conn.execute(f"UPDATE cases SET {cols} WHERE case_ref=?", list(kwargs.values()) + [case_ref])
        conn.commit()

def get_all_cases(status=None, limit=100):
    with get_conn() as conn:
        if status:
            rows = conn.execute("SELECT * FROM cases WHERE status=? ORDER BY updated_at DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cases ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def create_followup(case_ref, caller_number, call_id, intent, urgency, sentiment,
                    sla_deadline, callback_dt, transcript_summary,
                    agent_id=None, auto_resolved=False, auto_action=None):
    with get_conn() as conn:
        cur = conn.execute("""
        INSERT INTO followups
            (case_ref,caller_number,call_id,agent_id,intent,urgency,sentiment,
             sla_deadline,callback_dt,transcript_summary,auto_resolved,auto_action)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (case_ref, caller_number, call_id, agent_id, intent, urgency, sentiment,
         sla_deadline, callback_dt, transcript_summary, 1 if auto_resolved else 0, auto_action))
        conn.commit()
        return cur.lastrowid

def get_followups(status=None, limit=100):
    with get_conn() as conn:
        if status:
            rows = conn.execute("SELECT * FROM followups WHERE status=? ORDER BY callback_dt ASC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM followups ORDER BY callback_dt ASC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def get_overdue_followups():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM followups WHERE status='Pending' AND callback_dt <= ? ORDER BY callback_dt ASC", (now,)).fetchall()
    return [dict(r) for r in rows]

def update_followup_status(followup_id, status, notes=""):
    resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == "Resolved" else None
    with get_conn() as conn:
        conn.execute(
            "UPDATE followups SET status=?, resolved_at=?, resolution_notes=? WHERE id=?",
            (status, resolved_at, notes, followup_id)
        )
        conn.commit()
        # Also add resolution_notes column if it doesn't exist (migration safety)
        try:
            conn.execute("ALTER TABLE followups ADD COLUMN resolution_notes TEXT DEFAULT ''")
            conn.commit()
        except Exception:
            pass  # column already exists

def get_all_agents():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM agents WHERE is_active=1").fetchall()
    return [dict(r) for r in rows]

def get_agent_schedule(agent_id):
    """Returns dict of hour_slot -> count of pending callbacks."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT strftime('%Y-%m-%d %H', callback_dt) as slot, COUNT(*) as cnt
            FROM followups WHERE agent_id=? AND status='Pending' GROUP BY slot""", (agent_id,)).fetchall()
    return {r["slot"]: r["cnt"] for r in rows}

def get_analytics():
    with get_conn() as conn:
        stats = {
            "total_calls":       conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
            "active_calls":      conn.execute("SELECT COUNT(*) FROM calls WHERE status='active'").fetchone()[0],
            "total_cases":       conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
            "open_cases":        conn.execute("SELECT COUNT(*) FROM cases WHERE status='Open'").fetchone()[0],
            "pending_followups": conn.execute("SELECT COUNT(*) FROM followups WHERE status='Pending'").fetchone()[0],
            "overdue_followups": conn.execute("SELECT COUNT(*) FROM followups WHERE status='Pending' AND callback_dt <= datetime('now','localtime')").fetchone()[0],
            "auto_resolved":     conn.execute("SELECT COUNT(*) FROM followups WHERE auto_resolved=1").fetchone()[0],
            "intent_dist":    [dict(r) for r in conn.execute("SELECT intent, COUNT(*) as cnt FROM calls WHERE intent IS NOT NULL GROUP BY intent ORDER BY cnt DESC").fetchall()],
            "sentiment_dist": [dict(r) for r in conn.execute("SELECT sentiment, COUNT(*) as cnt FROM calls WHERE sentiment IS NOT NULL GROUP BY sentiment").fetchall()],
            "urgency_dist":   [dict(r) for r in conn.execute("SELECT urgency, COUNT(*) as cnt FROM calls WHERE urgency IS NOT NULL GROUP BY urgency").fetchall()],
            "daily_calls":    [dict(r) for r in conn.execute("SELECT date(started_at) as day, COUNT(*) as cnt FROM calls GROUP BY day ORDER BY day DESC LIMIT 7").fetchall()],
        }
    return stats

if __name__ == "__main__":
    init_db()
