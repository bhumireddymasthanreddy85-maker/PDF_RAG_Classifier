"""
app.py
------
PDF RAG Document Classifier — a Streamlit app.

Pipeline:
    Upload PDFs -> extract text -> chunk -> embed (local, free) ->
    store in FAISS -> classify each document -> retrieve relevant
    passages for a user query -> export results to CSV/Excel.

100% local & free: no paid APIs, no API keys. Embedding models come
from Sentence-Transformers (Hugging Face) and run on your own CPU/GPU.
"""

import io
import os

import pandas as pd
import streamlit as st

from utils.classifier import CATEGORY_PROTOTYPES, classify_text
from utils.embeddings import embed_text, embed_texts
from utils.generator import generate_answer
from utils.pdf_parser import extract_text
from utils.text_chunker import chunk_text
from utils.vector_store import FAISSVectorStore

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="PDF RAG Document Classifier", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = {}


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Embedding model (local, free)",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    index=0,
    help="Downloaded once from Hugging Face, then runs fully offline.",
)
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 1500, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 300, 50, step=10)
top_k = st.sidebar.slider("Top-K chunks to retrieve", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.caption("Categories: " + ", ".join(CATEGORY_PROTOTYPES.keys()))
st.sidebar.caption("100% local pipeline — no API keys, no paid services.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 PDF RAG Document Classifier")
st.write(
    "Upload PDF documents, automatically classify them into categories, and "
    "search their content using a local retrieval-augmented pipeline — all "
    "without any paid APIs."
)

# ---------------------------------------------------------------------------
# 1. Upload & process
# ---------------------------------------------------------------------------
st.header("1️⃣ Upload & Process PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

process_btn = st.button("🚀 Process Documents", type="primary", disabled=not uploaded_files)

if process_btn and uploaded_files:
    progress = st.progress(0, text="Starting...")

    results = []
    doc_texts = {}
    all_chunk_texts = []
    all_chunk_meta = []

    total = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(i / total, text=f"Extracting: {uploaded_file.name}")
        extraction = extract_text(uploaded_file)
        text = extraction["text"]
        doc_texts[uploaded_file.name] = text

        if not text:
            results.append(
                {
                    "filename": uploaded_file.name,
                    "category": "Unreadable",
                    "confidence": 0.0,
                    "num_chunks": 0,
                    "char_count": 0,
                    "backend_used": extraction["backend_used"],
                    "error": extraction["error"],
                }
            )
            continue

        progress.progress((i + 0.4) / total, text=f"Chunking: {uploaded_file.name}")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in chunks:
            all_chunk_texts.append(c)
            all_chunk_meta.append({"source": uploaded_file.name, "chunk_text": c})

        progress.progress((i + 0.7) / total, text=f"Classifying: {uploaded_file.name}")
        label, confidence, _all_scores = classify_text(text, model_name=model_name)

        results.append(
            {
                "filename": uploaded_file.name,
                "category": label,
                "confidence": round(confidence, 4),
                "num_chunks": len(chunks),
                "char_count": len(text),
                "backend_used": extraction["backend_used"],
                "error": None,
            }
        )

    if all_chunk_texts:
        progress.progress(0.9, text="Embedding chunks & building vector index...")
        chunk_embeddings = embed_texts(all_chunk_texts, model_name=model_name)
        vs = FAISSVectorStore(dim=chunk_embeddings.shape[1])
        vs.add(chunk_embeddings, all_chunk_meta)
        st.session_state.vector_store = vs
    else:
        st.session_state.vector_store = None

    st.session_state.results = results
    st.session_state.doc_texts = doc_texts
    progress.progress(1.0, text="Done!")
    st.success(f"Processed {total} document(s).")


# ---------------------------------------------------------------------------
# 2. Results table
# ---------------------------------------------------------------------------
if st.session_state.results:
    st.header("2️⃣ Classification Results")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(df["category"].value_counts())
    with col2:
        st.metric("Documents processed", len(df))
        readable = df[df["category"] != "Unreadable"]
        avg_conf = readable["confidence"].mean() if len(readable) else 0.0
        st.metric("Average confidence", f"{avg_conf:.2%}")


# ---------------------------------------------------------------------------
# 3. RAG-style retrieval / query
# ---------------------------------------------------------------------------
st.header("3️⃣ Ask Questions About Your Documents")
st.caption(
    "Semantic search finds the most relevant passages using local embeddings. "
    "Optionally, a small local model can also turn those passages into a "
    "written answer — both run on your own machine, no paid API involved."
)

generate_toggle = st.checkbox(
    "✨ Generate a written answer from the retrieved passages (downloads a small "
    "local model on first use — google/flan-t5-base, ~250MB)",
    value=False,
)

query = st.text_input("Enter a question or search phrase")
search_btn = st.button("🔍 Search", disabled=not query)

if search_btn and query:
    vs = st.session_state.vector_store
    if not vs or vs.size() == 0:
        st.warning("Please process at least one PDF with extractable text first.")
    else:
        query_vec = embed_text(query, model_name=model_name)
        hits = vs.search(query_vec, top_k=top_k)
        if not hits:
            st.info("No relevant results found.")
        else:
            if generate_toggle:
                with st.spinner("Generating answer from retrieved context..."):
                    context_chunks = [meta["chunk_text"] for meta, _score in hits]
                    answer = generate_answer(query, context_chunks)
                st.subheader("📝 Generated Answer")
                st.write(answer)
                st.caption(
                    "Generated locally from the passages below only — the model "
                    "cannot use any information outside your uploaded documents."
                )

            st.subheader("🔍 Retrieved Passages")
            for rank, (meta, score) in enumerate(hits, start=1):
                with st.expander(f"#{rank} — {meta['source']} (similarity: {score:.3f})"):
                    st.write(meta["chunk_text"])


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------
st.header("4️⃣ Export Results")

if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    excel_bytes = excel_buffer.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇️ Download Excel",
            data=excel_bytes,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.caption(f"A copy of the CSV is also saved to `outputs/results.csv` in the project folder.")
else:
    st.info("Process some documents first to enable export.")
