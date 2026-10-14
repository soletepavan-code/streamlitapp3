# Nova — Multimodal RAG Chat (100% free stack)

Claude-style chat UI · image understanding · PDF/DOCX/TXT/image RAG · persistent per-user chat memory.

**Stack:** Streamlit · Google Gemini free tier (chat, vision, embeddings) · SQLAlchemy (SQLite / free Postgres) · NumPy cosine search.

## Memory design
- Every message (and image) is persisted per user/session in the DB.
- Last 12 messages are sent verbatim; older ones are rolled into a running summary (cheap on free quota).
- Chats survive refresh and re-login. **Streamlit Cloud disk is ephemeral**, so set `DATABASE_URL` (free Supabase/Neon Postgres) for durable history.

## 1. Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add GEMINI_API_KEY
streamlit run app.py
```
Free key: https://aistudio.google.com/apikey

## 2. Push to GitHub
```bash
git init && git add . && git commit -m "Nova multimodal RAG"
git branch -M main
git remote add origin https://github.com/<you>/nova-rag.git
git push -u origin main
```
`.gitignore` already excludes `secrets.toml` and `data/` — never commit keys.

## 3. Deploy on Streamlit Cloud
1. share.streamlit.io → **Create app** → pick repo, branch `main`, file `app.py`.
2. **Advanced settings** → Python 3.11 or 3.12 → paste secrets (TOML):
```toml
GEMINI_API_KEY = "..."
DATABASE_URL = "postgresql://USER:PASS@HOST:5432/postgres"   # recommended
```
3. Deploy → share the `*.streamlit.app` URL.

## 4. Durable DB (free, 3 minutes)
Supabase → New project → Project Settings → Database → copy the **Session pooler** URI → set as `DATABASE_URL`. Tables are created automatically. URL-encode special characters in the password.

## Common errors
| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Package missing in `requirements.txt`; commit and reboot app |
| `GEMINI_API_KEY is not set` | Add to Settings → Secrets (TOML, quoted string) |
| `429 / RESOURCE_EXHAUSTED` | Free-tier limit; wait a minute (app already retries) |
| `404 model not found` | Set `CHAT_MODEL` / `EMBED_MODEL` secrets to a currently available Gemini model |
| Postgres connection fails | Use the pooler URI, encode password, check host/port |
| History vanished | You're on SQLite; set `DATABASE_URL` |
| Scanned PDF "no extractable text" | Upload pages as images (vision OCR) |

## Live test checklist
1. Create account → sign in. 2. Ask a general question (streaming). 3. Attach an image and ask about it.
4. Sidebar → Knowledge base → index a PDF → ask about it → check **Sources**.
5. Refresh the page, sign in → chat is still in Recents. 6. Delete a chat / document.

## Files
`app.py` UI · `llm.py` Gemini · `rag.py` ingest/retrieve · `memory.py` summary memory · `db.py` storage · `config.py` settings




Backend and UI tested; now writing the README and zipping the project.**Nova** is ready to run locally and deploy. The backend and database were smoke-tested, but I haven't run the Streamlit UI or called the Gemini API, so test it live before showing students. Everything is in `nova-rag.zip`, and the README has the run, GitHub, Streamlit Cloud and troubleshooting steps.

- **UI:** Claude-style chat, with a sidebar of recent chats, streaming replies, image attach in the chat box, and a Sources expander on answers.
- **Multimodal RAG:** you can index PDF, DOCX, TXT, MD, CSV and images (images are captioned and OCR'd by Gemini). Answers cite their sources.
- **Chat memory:**
  - Each user signs in with a username and password.
  - Chats and images are saved per user.
  - The last 12 messages are sent as-is, and older ones are rolled into a running summary.
  - There is also a daily message cap per user, to protect the free quota.
- **Free stack:** Gemini free tier for chat, vision and embeddings; SQLite or free Supabase Postgres for storage; no paid services.

**Set `DATABASE_URL` before the demo.** Streamlit Cloud's disk is wiped on reboot or redeploy, so with the default SQLite, chat history disappears. A free Supabase Postgres URL in the secrets keeps it. The README has the 3-minute setup.

**Check the model names.** If you get a "404 model not found" error, set `CHAT_MODEL` and `EMBED_MODEL` in the secrets to the Gemini models currently available. I used `gemini-2.5-flash` and `gemini-embedding-001` as defaults.

Smaller things to know: you'll need to sign in again after a browser refresh, though your chats are kept, and scanned PDFs have no text to extract, so upload their pages as images instead.