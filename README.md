# 📄 PDF RAG Document Classifier

A Streamlit app that uploads PDFs, extracts and chunks their text, embeds
the chunks locally, stores them in a FAISS vector index, retrieves
relevant passages for a query (RAG-style), and classifies each document
into a category — all using **free, local libraries only**. No API keys,
no paid services.

### Features
- Upload multiple PDFs
- Extract text using **pdfplumber**, **PyMuPDF**, and **PyPDF2** (automatic fallback chain)
- Chunk text with **LangChain**'s `RecursiveCharacterTextSplitter`
- Generate embeddings using **Sentence-Transformers** (free, local models — e.g. `all-MiniLM-L6-v2`)
- Store & search embeddings in a **FAISS** vector index
- Query documents with a local retrieval (RAG) pipeline — returns the most relevant passages
- Classify each document into one of five categories: **Job-related, Finance, Legal, Research, Spam**
  (via embedding similarity to category descriptions — no fine-tuned model or LLM API required)
- Export results to **CSV** and **Excel**

### Project Structure
```
pdf_rag_classifier/
│── app.py                  # Streamlit app (entry point)
│── requirements.txt
│── README.md
│── utils/
│    ├── pdf_parser.py      # Multi-backend PDF text extraction
│    ├── text_chunker.py    # LangChain-based text chunking
│    ├── embeddings.py      # Local Sentence-Transformers embeddings
│    ├── vector_store.py    # FAISS vector store wrapper
│    ├── classifier.py      # Embedding-similarity document classifier
│── data/
│    ├── sample.pdf         # Example PDF you can try the app with
│── outputs/
│    ├── results.csv        # Latest exported classification results
```

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### How to use
1. In the sidebar, choose an embedding model and chunking settings.
2. Upload one or more PDF files (try `data/sample.pdf` to start).
3. Click **Process Documents** — this extracts, chunks, embeds, and classifies each file.
4. Review the classification table and category breakdown chart.
5. Type a question in the **Ask Questions** box and click **Search** to retrieve the most relevant passages across all uploaded documents.
6. Click **Download CSV** / **Download Excel** to export the classification results (also saved to `outputs/results.csv`).

### Notes on "RAG" in this app
Section 3 has two parts:
1. **Retrieval** (always on): your query is embedded and matched against
   document chunks using cosine similarity in FAISS, and the top matching
   passages are shown directly under "Retrieved Passages."
2. **Generation** (optional, checkbox): if enabled, a small local, free
   model (`google/flan-t5-base`, via Hugging Face Transformers) reads those
   retrieved passages and writes an actual answer, shown under "Generated
   Answer." It's instructed to only use the retrieved context and to say so
   if the answer isn't in your documents, rather than making something up.
   This still runs 100% locally — the model downloads once (~250MB) and then
   works offline, no API key required.

Generation is off by default because the model download adds startup time
on first use; turn on the checkbox in Section 3 whenever you want a written
answer instead of raw excerpts.

### Notes on classification
Categories are matched using cosine similarity between a document's
embedding and embeddings of short hand-written descriptions of each
category (see `utils/classifier.py`). This is a lightweight, fully local
"zero-shot" approach — no labeled training data or paid classification
API required.

The **confidence score** shown is a softmax over those five similarity
scores, so it always sums to 100% across categories — it tells you how
much one category "won" relative to the others, not how certain the match
is in an absolute sense. If a document doesn't clearly resemble any
category (e.g. a resume with a lot of business/data vocabulary that
partially overlaps "Finance"), the app now labels it **"Uncertain"**
instead of forcing a low-confidence guess, based on the raw cosine
similarity falling below `MIN_CONFIDENCE_THRESHOLD` in `utils/classifier.py`.

For higher accuracy on your own document types, edit the
`CATEGORY_PROTOTYPES` dictionary in `utils/classifier.py` with more
representative keywords/descriptions, or add new categories.

### Requirements
All dependencies are free and open-source (see `requirements.txt`):
`streamlit`, `pdfplumber`, `PyPDF2`, `pymupdf`, `langchain`,
`langchain-text-splitters`, `sentence-transformers`, `faiss-cpu`,
`chromadb`, `torch`, `pandas`, `xlsxwriter`.

> First run will download the Sentence-Transformers embedding model
> (a few hundred MB) from Hugging Face. After that, everything runs
> fully offline.
