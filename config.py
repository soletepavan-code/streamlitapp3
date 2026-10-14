import os

import streamlit as st


def _get(key, default=None):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


GEMINI_API_KEY = _get("GEMINI_API_KEY")
CHAT_MODEL = _get("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = _get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(_get("EMBED_DIM", "768"))

_url = _get("DATABASE_URL", "sqlite:///data/app.db")
for _p in ("postgres://", "postgresql://"):
    if _url.startswith(_p):
        _url = "postgresql+psycopg2://" + _url[len(_p):]
        break
DATABASE_URL = _url

DAILY_MSG_LIMIT = int(_get("DAILY_MSG_LIMIT", "80"))   # protects the free quota, per user
WINDOW = int(_get("MEMORY_WINDOW", "12"))               # recent messages sent verbatim
SUMMARIZE_AFTER = int(_get("SUMMARIZE_AFTER", "24"))    # roll older messages into a summary
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SCORE = 0.35
