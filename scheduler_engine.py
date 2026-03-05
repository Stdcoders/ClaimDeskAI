from apscheduler.schedulers.background import BackgroundScheduler

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.start()
    return scheduler


def escalate_case(case_id):
    print(f"🚨 SLA BREACH: Case {case_id} requires escalation!")


def followup_case(case_id):
    print(f"📞 Follow-up Reminder for Case {case_id}")


def schedule_escalation(scheduler, case_id, run_time):

    scheduler.add_job(
        escalate_case,
        trigger="date",
        run_date=run_time,
        args=[case_id]
    )


def schedule_followup(scheduler, case_id, run_time):

    scheduler.add_job(
        followup_case,
        trigger="date",
        run_date=run_time,
        args=[case_id]
    )
