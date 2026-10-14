import io
import mimetypes
import re
import time

import streamlit as st
from PIL import Image

import config as cfg
import db
import llm
import memory
import rag

st.set_page_config(page_title="Nova", page_icon="✳️", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 780px; padding-top: 2rem; padding-bottom: 6rem;}
html, body, [class*="css"] {font-family: -apple-system, "Segoe UI", Inter, sans-serif;}
.hero {font-family: Georgia, "Times New Roman", serif; font-size: 2.4rem; text-align:center; margin: 18vh 0 .3rem; color:#3D3929;}
.sub {text-align:center; color:#8A8778; margin-bottom: 1.5rem;}
[data-testid="stChatMessage"] {background: transparent; padding: .4rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:#F0EEE6; border-radius:16px; padding:.7rem 1rem; margin-left:12%;}
[data-testid="stChatInput"] {border-radius: 20px; border:1px solid #E3E0D5; box-shadow: 0 2px 10px rgba(0,0,0,.05);}
section[data-testid="stSidebar"] {border-right: 1px solid #E8E5DA;}
section[data-testid="stSidebar"] .stButton>button {text-align:left; justify-content:flex-start; border:none;
  background:transparent; padding:.35rem .6rem; border-radius:8px; font-weight:400;}
section[data-testid="stSidebar"] .stButton>button:hover {background:#E6E3D8;}
section[data-testid="stSidebar"] .stButton>button[kind="primary"] {background:#E6E3D8; color:#3D3929;}
.brand {font-family: Georgia, serif; font-size:1.4rem; margin-bottom:.5rem;}
</style>
""", unsafe_allow_html=True)

SYSTEM = """You are Nova, a helpful, honest multimodal assistant. You can see images the user shares.
Answer in clear Markdown. If CONTEXT from the user's knowledge base is provided and relevant, use it and cite
sources inline as [1], [2]. If the context doesn't contain the answer, say so briefly, then answer from general
knowledge. Never invent citations."""

DOC_TYPES = ["pdf", "docx", "txt", "md", "csv"]
IMG_TYPES = ["png", "jpg", "jpeg", "webp"]


# ---------------- helpers ----------------
def prep_image(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "image/jpeg", buf.getvalue()


def ingest_files(uid, files):
    for f in files:
        mime = f.type or mimetypes.guess_type(f.name)[0] or ""
        try:
            n = rag.ingest(uid, f.name, f.getvalue(), mime)
            st.toast(f"Indexed {f.name} ({n} chunks)", icon="✅")
        except Exception as e:
            st.toast(f"{f.name}: {e}", icon="⚠️")


def friendly_error(e: Exception) -> str:
    s = str(e)
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "Free-tier rate limit reached. Please wait a minute and try again."
    if "API key" in s or "403" in s or "401" in s:
        return "The Gemini API key is missing or invalid. Check your secrets."
    return f"Something went wrong: {s[:200]}"


def render_message(m):
    with st.chat_message(m["role"], avatar="✳️" if m["role"] == "assistant" else None):
        for _, data in m["images"]:
            st.image(data, width=260)
        st.markdown(m["content"])
        render_sources(m["meta"].get("sources"))


def render_sources(src):
    if not src:
        return
    with st.expander(f"Sources ({len(src)})"):
        for i, s in enumerate(src, 1):
            pg = f", p.{s['page']}" if s.get("page") else ""
            st.markdown(f"**[{i}] {s['source']}{pg}** · relevance {s['score']:.2f}")
            st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))


# ---------------- auth ----------------
def auth_screen():
    st.markdown('<div class="hero">✳ Nova</div><div class="sub">Multimodal RAG assistant — chat, images, documents</div>',
                unsafe_allow_html=True)
    t1, t2 = st.tabs(["Sign in", "Create account"])
    with t1, st.form("login"):
        u, p = st.text_input("Username"), st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", use_container_width=True):
            uid = db.verify_user(u, p)
            if uid:
                st.session_state.update(uid=uid, username=u.strip().lower(), sid=None)
                st.rerun()
            st.error("Invalid username or password.")
    with t2, st.form("signup"):
        u, p = st.text_input("Choose a username"), st.text_input("Choose a password (8+ chars)", type="password")
        if st.form_submit_button("Create account", use_container_width=True):
            u = u.strip().lower()
            if not re.fullmatch(r"[a-z0-9_]{3,32}", u):
                st.error("Username: 3–32 chars, letters/numbers/underscore.")
            elif len(p) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                uid = db.create_user(u, p)
                if uid:
                    st.session_state.update(uid=uid, username=u, sid=None)
                    st.rerun()
                st.error("That username is taken.")


# ---------------- sidebar ----------------
def sidebar(uid):
    with st.sidebar:
        st.markdown('<div class="brand">✳ Nova</div>', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True):
            st.session_state.sid = None
            st.rerun()

        st.caption("Recents")
        for s in db.list_sessions(uid):
            active = s["id"] == st.session_state.get("sid")
            if st.button(s["title"] or "New chat", key=f"s_{s['id']}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.sid = s["id"]
                st.rerun()

        st.divider()
        with st.expander("📚 Knowledge base"):
            with st.form("kb", clear_on_submit=True):
                files = st.file_uploader("Documents or images", type=DOC_TYPES + IMG_TYPES,
                                         accept_multiple_files=True)
                if st.form_submit_button("Index files", use_container_width=True) and files:
                    with st.spinner("Indexing…"):
                        ingest_files(uid, files)
                    st.rerun()
            for d in db.list_documents(uid):
                c1, c2 = st.columns([5, 1])
                c1.caption(f"{d['name']} · {d['n_chunks']}")
                if c2.button("✕", key=f"d_{d['id']}"):
                    db.delete_document(uid, d["id"])
                    st.rerun()

        if st.session_state.get("sid") and st.button("🗑  Delete this chat", use_container_width=True):
            db.delete_session(st.session_state.sid, uid)
            st.session_state.sid = None
            st.rerun()
        st.caption(f"Signed in as **{st.session_state.username}**")
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# ---------------- chat ----------------
def chat(uid):
    sid = st.session_state.get("sid")
    msgs = db.get_messages(sid) if sid else []

    if not msgs:
        st.markdown(f'<div class="hero">✳ Hello, {st.session_state.username}</div>'
                    '<div class="sub">Ask anything, attach an image, or add documents to your knowledge base.</div>',
                    unsafe_allow_html=True)
    for m in msgs:
        render_message(m)

    prompt = st.chat_input("How can I help you today?", accept_file="multiple", file_type=IMG_TYPES + DOC_TYPES)
    if not prompt:
        return

    text = (prompt.text or "").strip()
    files = prompt.files or []
    imgs = [prep_image(f.getvalue()) for f in files if (f.type or "").startswith("image/")]
    docs = [f for f in files if not (f.type or "").startswith("image/")]
    if docs:
        with st.spinner("Indexing documents…"):
            ingest_files(uid, docs)
    if not text and not imgs:
        return
    text = text or "What's in this image?"

    if db.count_user_messages_since(uid, time.time() - 86400) >= cfg.DAILY_MSG_LIMIT:
        st.warning(f"Daily limit of {cfg.DAILY_MSG_LIMIT} messages reached (free-tier protection). Try again tomorrow.")
        return

    first = sid is None
    if first:
        sid = st.session_state.sid = db.new_session(uid)
    summary, history = memory.build_context(sid) if not first else ("", [])
    db.add_message(sid, "user", text, images=imgs)
    render_message(dict(role="user", content=text, images=imgs, meta={}))

    with st.chat_message("assistant", avatar="✳️"):
        sources = []
        try:
            q = text if len(text) > 40 or not history else next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), "") + " " + text
            sources = rag.retrieve(uid, q)
        except Exception:
            pass
        system = SYSTEM
        if summary:
            system += f"\n\nCONVERSATION SUMMARY (earlier messages):\n{summary}"
        if sources:
            system += "\n\nCONTEXT:\n" + "\n\n".join(
                f"[{i}] ({s['source']}{', p.' + str(s['page']) if s['page'] else ''}) {s['text']}"
                for i, s in enumerate(sources, 1))
        try:
            answer = st.write_stream(llm.stream_chat(system, history, text, imgs))
            render_sources(sources)
            db.add_message(sid, "assistant", answer, meta={"sources": sources})
        except Exception as e:
            st.error(friendly_error(e))
            answer = None

    if answer:
        if first:
            db.rename_session(sid, (text[:48] + "…") if len(text) > 48 else text)
        try:
            memory.maybe_summarize(sid)
        except Exception:
            pass
        if first:
            st.rerun()


# ---------------- main ----------------
if not cfg.GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in .streamlit/secrets.toml (local) or App → Settings → Secrets (cloud).")
    st.stop()
db.engine()
if "uid" not in st.session_state:
    auth_screen()
else:
    sidebar(st.session_state.uid)
    chat(st.session_state.uid)
