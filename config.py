"""Shared settings for every Project B script. Change values here, not inside the scripts."""
from pathlib import Path

DATA_DIR    = Path("data")                      # scraped articles and search index (not committed to GitHub)
RESULTS_DIR = Path("results")                   # evaluation results (committed, as evidence)
ARTICLES_FILE = DATA_DIR / "articles.jsonl"
CHUNKS_FILE   = DATA_DIR / "chunks.jsonl"

# Chunking: passages of roughly 80-180 words, built from whole paragraphs
CHUNK_MAX_WORDS = 180
CHUNK_MIN_WORDS = 80

# Embedding options: "gemini" (Google API) or "local" (Sentence Transformers on this computer)
EMBEDDING_BACKEND  = "gemini"                   # chosen by compare_retrieval.py (hit rate 1.0 vs 0.917); "local" is the backup
GEMINI_EMBED_MODEL = "gemini-embedding-001"
LOCAL_EMBED_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_FILES = {"gemini": DATA_DIR / "embeddings_gemini.npy",
                   "local":  DATA_DIR / "embeddings_local.npy"}
# Each embedding file has a small "fingerprint" file, so passages and embeddings can never be mismatched
EMBEDDING_META_FILES = {backend: path.with_suffix(".json") for backend, path in EMBEDDING_FILES.items()}

# Answering
# Answer models, tried in order. On the free tier, daily limits are counted per project AND per model
# (our own error message showed "GenerateRequestsPerDayPerProjectPerModel", limit 20 for gemini-3-flash),
# so if one model's allowance is used up, or it isn't available to this key, the next one is tried.
# Limits and availability vary by account and change over time: check them in Google AI Studio.
GEMINI_CHAT_MODELS = [
    "gemini-3.5-flash",            # first choice; the evaluations also recorded answers from gemini-3.5-flash-lite
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
]
TOP_K = 5                                       # passages given to the model for each question

# Guardrail layer 1: if even the best-matching passage scores below this, decline without calling Gemini.
# Chosen on the DEVELOPMENT questions (compare_retrieval.py / evaluate.py), then frozen before final_check.py:
# the weakest in-scope question scored 0.702 and clearly unrelated ones 0.59-0.61.
# First set at 0.68, lowered to 0.65 after evaluate.py showed layer 2 (the prompt) reliably declines
# near-topic questions such as other states' elections, so a lower cut-off avoids refusing real questions
# that are phrased differently. Backup (local) value set the same way (weakest in-scope 0.454).
SIMILARITY_THRESHOLD = {"gemini": 0.65, "local": 0.43}
