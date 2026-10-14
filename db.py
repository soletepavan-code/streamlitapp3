"""Persistence layer (SQLAlchemy Core). SQLite by default, Postgres via DATABASE_URL."""
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid

import numpy as np
from sqlalchemy import (Column, Float, ForeignKey, Integer, LargeBinary, MetaData, String,
                        Table, Text, create_engine, delete, event, func, insert, select, update)
from sqlalchemy.exc import IntegrityError

import config as cfg

metadata = MetaData()

users = Table("users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("salt", LargeBinary, nullable=False),
    Column("pw_hash", LargeBinary, nullable=False),
    Column("created_at", Float))

sessions = Table("sessions", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), index=True, nullable=False),
    Column("title", String(200), default="New chat"),
    Column("summary", Text, default=""),
    Column("summarized_upto", Integer, default=0),
    Column("created_at", Float),
    Column("updated_at", Float, index=True))

messages = Table("messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(32), ForeignKey("sessions.id"), index=True, nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, default=""),
    Column("meta", Text, default="{}"),
    Column("created_at", Float))

attachments = Table("attachments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("message_id", Integer, ForeignKey("messages.id"), index=True, nullable=False),
    Column("mime", String(64)),
    Column("data", LargeBinary))

documents = Table("documents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), index=True, nullable=False),
    Column("name", String(255)),
    Column("kind", String(16)),
    Column("n_chunks", Integer),
    Column("created_at", Float))

chunks = Table("chunks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("doc_id", Integer, ForeignKey("documents.id"), index=True, nullable=False),
    Column("user_id", Integer, index=True, nullable=False),
    Column("page", Integer),
    Column("text", Text),
    Column("embedding", LargeBinary))

_engine = None


def engine():
    global _engine
    if _engine is None:
        url = cfg.DATABASE_URL
        if url.startswith("sqlite"):
            os.makedirs("data", exist_ok=True)
            _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

            @event.listens_for(_engine, "connect")
            def _pragmas(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        else:
            _engine = create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=300)
        metadata.create_all(_engine)
    return _engine


# ---------- auth ----------
def _hash(pw: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)


def create_user(username: str, password: str):
    salt = secrets.token_bytes(16)
    try:
        with engine().begin() as c:
            r = c.execute(insert(users).values(username=username, salt=salt,
                                               pw_hash=_hash(password, salt), created_at=time.time()))
            return r.inserted_primary_key[0]
    except IntegrityError:
        return None


def verify_user(username: str, password: str):
    with engine().connect() as c:
        r = c.execute(select(users).where(users.c.username == username.strip().lower())).mappings().first()
    if r and hmac.compare_digest(r["pw_hash"], _hash(password, r["salt"])):
        return r["id"]
    return None


# ---------- sessions & messages ----------
def new_session(uid: int) -> str:
    sid, now = uuid.uuid4().hex, time.time()
    with engine().begin() as c:
        c.execute(insert(sessions).values(id=sid, user_id=uid, title="New chat", summary="",
                                          summarized_upto=0, created_at=now, updated_at=now))
    return sid


def list_sessions(uid: int, limit: int = 40):
    with engine().connect() as c:
        return c.execute(select(sessions.c.id, sessions.c.title).where(sessions.c.user_id == uid)
                         .order_by(sessions.c.updated_at.desc()).limit(limit)).mappings().all()


def get_session(sid: str, uid: int):
    with engine().connect() as c:
        return c.execute(select(sessions).where(sessions.c.id == sid, sessions.c.user_id == uid)).mappings().first()


def rename_session(sid: str, title: str):
    with engine().begin() as c:
        c.execute(update(sessions).where(sessions.c.id == sid).values(title=title[:60]))


def delete_session(sid: str, uid: int):
    if not get_session(sid, uid):
        return
    with engine().begin() as c:
        mids = select(messages.c.id).where(messages.c.session_id == sid)
        c.execute(delete(attachments).where(attachments.c.message_id.in_(mids)))
        c.execute(delete(messages).where(messages.c.session_id == sid))
        c.execute(delete(sessions).where(sessions.c.id == sid))


def add_message(sid: str, role: str, content: str, images=None, meta=None) -> int:
    now = time.time()
    with engine().begin() as c:
        mid = c.execute(insert(messages).values(session_id=sid, role=role, content=content,
                                                meta=json.dumps(meta or {}), created_at=now)).inserted_primary_key[0]
        for mime, data in (images or []):
            c.execute(insert(attachments).values(message_id=mid, mime=mime, data=data))
        c.execute(update(sessions).where(sessions.c.id == sid).values(updated_at=now))
    return mid


def get_messages(sid: str):
    with engine().connect() as c:
        rows = c.execute(select(messages).where(messages.c.session_id == sid).order_by(messages.c.id)).mappings().all()
        att = {}
        ids = [r["id"] for r in rows]
        if ids:
            for a in c.execute(select(attachments).where(attachments.c.message_id.in_(ids))
                               .order_by(attachments.c.id)).mappings():
                att.setdefault(a["message_id"], []).append((a["mime"], bytes(a["data"])))
    return [dict(id=r["id"], role=r["role"], content=r["content"] or "",
                 meta=json.loads(r["meta"] or "{}"), images=att.get(r["id"], [])) for r in rows]


def set_summary(sid: str, summary: str, upto_id: int):
    with engine().begin() as c:
        c.execute(update(sessions).where(sessions.c.id == sid).values(summary=summary, summarized_upto=upto_id))


def count_user_messages_since(uid: int, ts: float) -> int:
    j = messages.join(sessions, sessions.c.id == messages.c.session_id)
    with engine().connect() as c:
        return c.execute(select(func.count()).select_from(j).where(
            sessions.c.user_id == uid, messages.c.role == "user", messages.c.created_at > ts)).scalar() or 0


# ---------- knowledge base ----------
def add_document(uid: int, name: str, kind: str, parts, embs: np.ndarray) -> int:
    """parts: list of (page, text)."""
    with engine().begin() as c:
        did = c.execute(insert(documents).values(user_id=uid, name=name, kind=kind, n_chunks=len(parts),
                                                 created_at=time.time())).inserted_primary_key[0]
        c.execute(insert(chunks), [dict(doc_id=did, user_id=uid, page=p, text=t,
                                        embedding=e.astype(np.float32).tobytes())
                                   for (p, t), e in zip(parts, embs)])
    return did


def list_documents(uid: int):
    with engine().connect() as c:
        return c.execute(select(documents.c.id, documents.c.name, documents.c.kind, documents.c.n_chunks)
                         .where(documents.c.user_id == uid).order_by(documents.c.id.desc())).mappings().all()


def delete_document(uid: int, doc_id: int):
    with engine().begin() as c:
        c.execute(delete(chunks).where(chunks.c.doc_id == doc_id, chunks.c.user_id == uid))
        c.execute(delete(documents).where(documents.c.id == doc_id, documents.c.user_id == uid))


def load_chunks(uid: int):
    q = (select(chunks.c.text, chunks.c.page, chunks.c.embedding, documents.c.name)
         .join(documents, documents.c.id == chunks.c.doc_id).where(chunks.c.user_id == uid))
    with engine().connect() as c:
        rows = c.execute(q).mappings().all()
    if not rows:
        return [], None
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return [dict(text=r["text"], page=r["page"], source=r["name"]) for r in rows], mat
