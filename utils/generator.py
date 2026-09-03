"""
generator.py
------------
Optional answer-generation step that turns retrieved chunks into an actual
written answer, using a small local, free, open-weight text2text model
(google/flan-t5-base) via Hugging Face Transformers. Runs entirely on your
own CPU — no API key, no paid service, no internet after the first
one-time model download.

This is what turns the app from pure retrieval ("here are some passages
that might be relevant") into real Retrieval-Augmented Generation
("here is an answer, built from those passages").
"""

from typing import List

DEFAULT_GEN_MODEL = "google/flan-t5-base"

_model_cache = {}


def load_generator(model_name: str = DEFAULT_GEN_MODEL):
    """Load (and cache) a local seq2seq model + tokenizer.

    We load the model and tokenizer directly (instead of using
    transformers.pipeline) because pipeline task names have changed across
    transformers versions, and 'text2text-generation' isn't registered in
    every release. Calling model.generate() directly avoids that entirely.
    """
    if model_name not in _model_cache:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


def generate_answer(
    question: str,
    context_chunks: List[str],
    model_name: str = DEFAULT_GEN_MODEL,
    max_new_tokens: int = 200,
) -> str:
    """
    Generate a short answer to `question`, grounded only in `context_chunks`.

    If the retrieved context doesn't actually contain the answer, the model
    is instructed to say so rather than invent one.
    """
    if not context_chunks:
        return "No relevant context was retrieved, so no answer can be generated."

    context = "\n\n".join(context_chunks)
    # Keep the prompt within a reasonable length for flan-t5's context window.
    context = context[:3000]

    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say 'The documents do not "
        "contain enough information to answer this.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    tokenizer, model = load_generator(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()
