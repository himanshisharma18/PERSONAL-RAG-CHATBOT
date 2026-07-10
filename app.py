import os
import numpy as np
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

st.set_page_config(page_title="Personal Knowledge Chatbot", layout="wide")

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
TOP_K = 4

# ---------------- LLM API config ----------------
# Get a free key at https://console.groq.com (no credit card needed)
API_BASE_URL = os.environ.get("LLM_API_BASE", "https://api.groq.com/openai/v1/chat/completions")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL_NAME = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def extract_text(file) -> str:
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, embedder, k=TOP_K):
    q_emb = embedder.encode([query]).astype("float32")
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]


def call_llm(query, context):
    if not API_KEY:
        return ("⚠️ No LLM_API_KEY set. Get a free key at console.groq.com and set it as an "
                "environment variable before running: `export LLM_API_KEY=your_key_here`")

    prompt = f"""Answer the question using ONLY the context below. If the answer is not contained in the context, say "I don't have that information in your documents."

Context:
{context}

Question: {query}

Answer:"""

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers strictly from the given context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


# ---------------- UI ----------------
st.title("📚 Personal Knowledge Chatbot (RAG)")
st.caption("Upload your notes/PDFs, build a knowledge base, then ask questions grounded in your own documents.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.history = []

embedder = load_embedder()

with st.sidebar:
    st.header("1. Build Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs or .txt notes", accept_multiple_files=True, type=["pdf", "txt"])
    if st.button("Build / Rebuild Index"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            all_chunks = []
            for f in uploaded_files:
                text = extract_text(f)
                all_chunks.extend(chunk_text(text))
            if not all_chunks:
                st.error("No extractable text found in uploaded files.")
            else:
                st.session_state.chunks = all_chunks
                st.session_state.index = build_index(all_chunks, embedder)
                st.session_state.history = []
                st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Key set: " + ("✅" if API_KEY else "❌ (set LLM_API_KEY)"))

st.header("2. Chat")
if st.session_state.index is None:
    st.info("Upload documents and click 'Build / Rebuild Index' to begin.")
else:
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

    query = st.chat_input("Ask something about your documents...")
    if query:
        st.session_state.history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)

        retrieved = retrieve(query, st.session_state.index, st.session_state.chunks, embedder)
        context = "\n\n---\n\n".join(retrieved)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_llm(query, context)
                st.write(answer)
                with st.expander("Sources used for this answer"):
                    for i, r in enumerate(retrieved):
                        st.markdown(f"**Chunk {i + 1}:** {r[:300]}...")
        st.session_state.history.append(("assistant", answer))
