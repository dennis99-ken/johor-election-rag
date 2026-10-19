"""Turning text into embeddings (Gemini or Sentence Transformers) and finding the closest passages."""
import hashlib
import json
import time

import numpy as np

import config

_gemini_client = None
_local_model = None


def _gemini():
    """One shared Gemini client; it reads the GEMINI_API_KEY environment variable."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client()
    return _gemini_client


def call_with_retry(function, what, attempts=4):
    """Call Gemini; if it says 'too many requests' or 'busy', wait and try again (free-tier per-minute limits).
    A DAILY limit can't be fixed by waiting, so that error is passed straight back to the caller."""
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as error:
            message = str(error)
            if "PerDay" in message:
                raise
            if any(code in message for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")) and attempt < attempts:
                wait = 15 * attempt
                print(f"   Gemini limit reached or busy ({what}); waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise


def normalise(vectors):
    """Scale every vector to length 1, so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def embed_gemini(texts, task):
    """task: "RETRIEVAL_DOCUMENT" for passages, "RETRIEVAL_QUERY" for questions."""
    from google.genai import types
    vectors = []
    for start in range(0, len(texts), 50):                  # send passages in batches of 50
        batch = texts[start:start + 50]
        result = call_with_retry(
            lambda: _gemini().models.embed_content(model=config.GEMINI_EMBED_MODEL, contents=batch,
                                                   config=types.EmbedContentConfig(task_type=task)),
            "embedding")
        vectors.extend(e.values for e in result.embeddings)
    return normalise(vectors)


def embed_local(texts):
    """Sentence Transformers model running on this computer (downloaded once, about 90 MB)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return normalise(_local_model.encode(texts, show_progress_bar=False))


def embed(texts, backend, kind):
    """kind: "passage" or "question"."""
    if backend == "gemini":
        return embed_gemini(texts, "RETRIEVAL_DOCUMENT" if kind == "passage" else "RETRIEVAL_QUERY")
    return embed_local(texts)


def load_chunks():
    with config.CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fingerprint(chunks):
    """A short code that changes whenever any passage, or the passage order, changes."""
    return hashlib.sha256("\x1e".join(c["embed_text"] for c in chunks).encode("utf-8")).hexdigest()[:16]


def load_embeddings(backend, chunks):
    """Load one backend's embeddings, refusing them if they were built for different passages."""
    vectors_file, meta_file = config.EMBEDDING_FILES[backend], config.EMBEDDING_META_FILES[backend]
    if not vectors_file.exists() or not meta_file.exists():
        raise RuntimeError(f"No complete '{backend}' index found. Run: python build_index.py")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    vectors = np.load(vectors_file)
    expected_model = config.GEMINI_EMBED_MODEL if backend == "gemini" else config.LOCAL_EMBED_MODEL
    if meta.get("model") != expected_model:
        raise RuntimeError(f"The '{backend}' index was built with {meta.get('model')}, but config.py now uses "
                           f"{expected_model}. Run: python build_index.py")
    if (vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0
            or vectors.shape[1] != meta.get("dimensions")):
        raise RuntimeError(f"The '{backend}' embeddings file is damaged or incomplete. Run: python build_index.py")
    if (meta.get("fingerprint") != fingerprint(chunks) or vectors.shape[0] != len(chunks)
            or not np.isfinite(vectors).all()):
        raise RuntimeError(f"The '{backend}' embeddings don't match the current passages. Run: python build_index.py")
    return vectors


def search(question_vector, passage_vectors, top_k=config.TOP_K):
    """Cosine similarity with every passage (NumPy); returns the indices and scores of the best matches."""
    scores = passage_vectors @ question_vector
    best = np.argsort(-scores)[:top_k]
    return best, scores[best]
