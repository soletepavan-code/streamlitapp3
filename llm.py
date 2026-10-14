"""Gemini wrapper: chat (streaming, multimodal), embeddings, vision, one-shot generation."""
import time

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

import config as cfg


@st.cache_resource
def client():
    return genai.Client(api_key=cfg.GEMINI_API_KEY)


def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE"))


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or not _retryable(e):
                raise
            time.sleep(2 ** i)


def embed(texts, task="RETRIEVAL_DOCUMENT") -> np.ndarray:
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        r = _retry(lambda: client().models.embed_content(
            model=cfg.EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=cfg.EMBED_DIM)))
        out += [e.values for e in r.embeddings]
    arr = np.array(out, dtype=np.float32)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def generate(prompt: str, images=None, temperature=0.2) -> str:
    parts = [types.Part.from_text(text=prompt)]
    for mime, data in (images or []):
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    r = _retry(lambda: client().models.generate_content(
        model=cfg.CHAT_MODEL, contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(temperature=temperature)))
    return (r.text or "").strip()


def describe_image(data: bytes, mime: str) -> str:
    return generate("Describe this image in detail for search indexing. Transcribe all visible text "
                    "(OCR), and describe objects, charts, tables, diagrams and their meaning.",
                    images=[(mime, data)])


def stream_chat(system: str, history: list, user_text: str, images: list):
    """history: [{role, content, images:[(mime,bytes)]}]; yields text chunks."""
    contents = []
    for m in history:
        parts = [types.Part.from_text(text=m["content"] or "(image)")]
        parts += [types.Part.from_bytes(data=d, mime_type=mt) for mt, d in m["images"]]
        contents.append(types.Content(role="user" if m["role"] == "user" else "model", parts=parts))
    while contents and contents[0].role != "user":
        contents.pop(0)
    parts = [types.Part.from_text(text=user_text)]
    parts += [types.Part.from_bytes(data=d, mime_type=mt) for mt, d in images]
    contents.append(types.Content(role="user", parts=parts))
    conf = types.GenerateContentConfig(system_instruction=system, temperature=0.4)

    for attempt in range(3):
        started = False
        try:
            for chunk in client().models.generate_content_stream(model=cfg.CHAT_MODEL, contents=contents, config=conf):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except Exception as e:
            if started or attempt == 2 or not _retryable(e):
                raise
            time.sleep(2 ** attempt)
