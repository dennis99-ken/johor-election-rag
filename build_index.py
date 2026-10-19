"""
Step 2a - Build the search index:
  1. split each article into passages (chunks) of whole paragraphs, keeping title, date, author and link;
  2. turn every passage into an embedding with BOTH options (Gemini and Sentence Transformers);
  3. save everything in data/ for searching.

Run:  python build_index.py
"""
import json
import time

import numpy as np

import config
from retrieval import embed, fingerprint


def chunk_article(article):
    """Group consecutive paragraphs into passages of about CHUNK_MIN_WORDS-CHUNK_MAX_WORDS words."""
    chunks, current = [], []
    for paragraph in article["text"].split("\n"):
        words_so_far = sum(len(p.split()) for p in current)
        if current and words_so_far + len(paragraph.split()) > config.CHUNK_MAX_WORDS and words_so_far >= config.CHUNK_MIN_WORDS:
            chunks.append(current)
            current = []
        current.append(paragraph)
    if current:
        if chunks and sum(len(p.split()) for p in current) < 40:   # merge a very short tail into the previous passage
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return ["\n".join(c) for c in chunks]


def main():
    with config.ARTICLES_FILE.open(encoding="utf-8") as f:
        articles = [json.loads(line) for line in f if line.strip()]
    articles = [a for a in articles if a.get("text", "").strip()]
    if not articles:
        raise SystemExit("No articles with text found in data/articles.jsonl. Run: python scrape.py")

    chunks = []
    for article in articles:
        for number, text in enumerate(chunk_article(article)):
            chunks.append({
                "chunk_id": len(chunks), "url": article["url"], "title": article["title"],
                "published": article["published"], "author": article["author"], "section": article["section"],
                "part": number + 1, "text": text,
                # what gets embedded: the passage plus its title and date, so each passage keeps its context
                "embed_text": f"{article['title']} ({article['published'][:10]})\n{text}",
            })
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    sizes = [len(c["text"].split()) for c in chunks]
    print(f"{len(articles)} articles -> {len(chunks)} passages "
          f"(words per passage: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)})")

    texts = [c["embed_text"] for c in chunks]
    print_code = fingerprint(chunks)
    for backend in ["gemini", "local"]:
        # Remove this backend's old files first, so old embeddings can never be paired with new passages
        for old_file in (config.EMBEDDING_FILES[backend], config.EMBEDDING_META_FILES[backend]):
            old_file.unlink(missing_ok=True)
        started = time.time()
        try:
            vectors = embed(texts, backend, kind="passage")
        except Exception as error:      # one option failing must not stop the other
            print(f"[{backend}] FAILED (this option is unavailable until a successful rebuild): {str(error)[:300]}")
            continue
        np.save(config.EMBEDDING_FILES[backend], vectors)
        config.EMBEDDING_META_FILES[backend].write_text(json.dumps({
            "fingerprint": print_code, "passages": len(chunks), "dimensions": int(vectors.shape[1]),
            "model": config.GEMINI_EMBED_MODEL if backend == "gemini" else config.LOCAL_EMBED_MODEL,
        }, indent=2), encoding="utf-8")
        print(f"[{backend}] {vectors.shape[0]} passages x {vectors.shape[1]} numbers each, "
              f"{time.time() - started:.1f}s -> {config.EMBEDDING_FILES[backend]}")


if __name__ == "__main__":
    main()
