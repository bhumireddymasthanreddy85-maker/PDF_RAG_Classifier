"""
classifier.py
-------------
Zero-shot-style document classifier that works by comparing a document's
embedding to embeddings of hand-written category "prototype" descriptions,
using cosine similarity. This avoids needing any paid classification API
or a fine-tuned model — it only relies on the same free local
Sentence-Transformers embeddings used for retrieval.

Categories: Job-related, Finance, Legal, Research, Spam
"""

from typing import Dict, Tuple

import numpy as np

from utils.embeddings import embed_text, embed_texts

CATEGORY_PROTOTYPES = {
    "Job-related": (
        "resume curriculum vitae CV professional experience work history education "
        "skills summary objective employment history job title company employer "
        "years of experience technical skills certifications projects developer "
        "engineer manager analyst references cover letter job application job "
        "posting interview hiring recruitment career position candidate applicant"
    ),
    "Finance": (
        "invoice bank statement financial report balance sheet income tax return "
        "budget accounting ledger payment receipt investment portfolio expense "
        "revenue profit and loss cash flow audit financial statement transaction "
        "credit debit loan interest rate stock shares dividend"
    ),
    "Legal": (
        "contract agreement terms and conditions legal notice lawsuit court filing "
        "clause liability jurisdiction compliance regulation statute policy attorney "
        "plaintiff defendant witness affidavit indemnification governing law "
        "confidentiality non-disclosure agreement party hereby whereas"
    ),
    "Research": (
        "research paper abstract methodology experiment results academic study "
        "hypothesis literature review citation journal dataset analysis conclusion "
        "peer reviewed findings statistical significance sample size survey "
        "university professor thesis dissertation publication"
    ),
    "Spam": (
        "congratulations you have won click here free prize claim now limited time "
        "offer act now guaranteed unsubscribe advertisement promotion urgent winner "
        "risk free no obligation lottery cash prize verify your account suspicious link"
    ),
}

# Minimum softmax score to trust a prediction. Below this, the document
# doesn't clearly resemble any category and we label it "Uncertain" instead
# of forcing a low-confidence guess.
MIN_CONFIDENCE_THRESHOLD = 0.30

# Cache prototype embeddings per model so we only compute them once.
_prototype_cache: Dict[str, Dict[str, np.ndarray]] = {}


def _get_prototype_embeddings(model_name: str) -> Dict[str, np.ndarray]:
    if model_name not in _prototype_cache:
        labels = list(CATEGORY_PROTOTYPES.keys())
        descriptions = list(CATEGORY_PROTOTYPES.values())
        vectors = embed_texts(descriptions, model_name=model_name)
        _prototype_cache[model_name] = dict(zip(labels, vectors))
    return _prototype_cache[model_name]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def classify_text(
    text: str, model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify a document's text into one of the predefined categories.

    Returns
    -------
    (predicted_label, confidence, all_scores) where all_scores maps every
    category to a pseudo-probability (softmax over cosine similarities).
    """
    if not text or not text.strip():
        return "Unknown", 0.0, {}

    # Cap length for speed; the first few thousand characters are usually
    # representative enough of a document's overall topic.
    doc_vector = embed_text(text[:5000], model_name=model_name)
    prototypes = _get_prototype_embeddings(model_name)

    scores = {label: _cosine_sim(doc_vector, vec) for label, vec in prototypes.items()}

    values = np.array(list(scores.values()))
    # Temperature scaling sharpens separation between close scores.
    exp_values = np.exp((values - values.max()) * 8)
    probs = exp_values / exp_values.sum()
    prob_scores = dict(zip(scores.keys(), probs.tolist()))

    predicted_label = max(prob_scores, key=prob_scores.get)
    confidence = prob_scores[predicted_label]

    # Softmax always sums to 100%, so it will always "pick a winner" even
    # when a document doesn't resemble any category well. Guard against
    # over-confident-looking mislabels by falling back to "Uncertain" when
    # the raw cosine similarity (not the softmax-inflated score) is weak.
    top_raw_score = scores[predicted_label]
    if top_raw_score < MIN_CONFIDENCE_THRESHOLD:
        return "Uncertain", confidence, prob_scores

    return predicted_label, confidence, prob_scores
