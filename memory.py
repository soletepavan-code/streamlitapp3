"""Conversation memory: persisted history + sliding window + rolling summary."""
import config as cfg
import db
import llm


def build_context(sid: str):
    s = db.get_session(sid, _uid(sid))
    msgs = [m for m in db.get_messages(sid) if m["id"] > (s["summarized_upto"] or 0)]
    return (s["summary"] or ""), msgs[-cfg.WINDOW:]


def _uid(sid):
    from sqlalchemy import select
    with db.engine().connect() as c:
        return c.execute(select(db.sessions.c.user_id).where(db.sessions.c.id == sid)).scalar()


def maybe_summarize(sid: str):
    s = db.get_session(sid, _uid(sid))
    pending = [m for m in db.get_messages(sid) if m["id"] > (s["summarized_upto"] or 0)]
    if len(pending) <= cfg.SUMMARIZE_AFTER:
        return
    old = pending[:-cfg.WINDOW]
    transcript = "\n".join(f'{m["role"]}: {m["content"][:1500]}' for m in old)
    summary = llm.generate(
        "Update the running conversation summary. Keep durable facts, user goals and preferences, "
        "decisions, and open tasks. Max 250 words, bullet points.\n\n"
        f"EXISTING SUMMARY:\n{s['summary'] or '(none)'}\n\nNEW MESSAGES:\n{transcript}")
    if summary:
        db.set_summary(sid, summary, old[-1]["id"])
