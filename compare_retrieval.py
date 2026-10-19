"""
Step 2b - Compare the two embedding options (and check FAISS) on test questions.

For each option it reports:
  * Hit rate @5 (article level): for in-scope questions, is a passage from the right article among the top 5?
    (This checks the article, not that the exact passage contains the answer.)
  * MRR @5: how high the first correct passage is ranked (1 = always first).
  * Separation: do out-of-scope questions score clearly LOWER than in-scope ones?
    This is what lets the chatbot safely say "I don't know" instead of guessing.
It also checks that FAISS returns the same results as our NumPy search.

These are DEVELOPMENT checks: the same questions were used to choose the embedding option and the cut-off.
Run:  python compare_retrieval.py      (results are saved in results/)
"""
import numpy as np
import pandas as pd

import config
from retrieval import embed, load_chunks, load_embeddings, search

# In-scope questions, each with words that appear in the link(s) of the article(s) that answer it
IN_SCOPE = [
    ("How many seats did Barisan Nasional win in the 2026 Johor state election?",
     ["explore-the-johor-poll-results", "barisan-reps-urged", "sworn-in-as-johor-mentri-besar", "ahmad-zahid-congratulates",
      "pakatan-suffers-crushing-defeat", "major-setback-for-pakatan", "bersatu-to-study", "onn-hafiz-pledges", "bn-has-won-over-40-seats"]),
    ("When was nomination day for the Johor polls?", ["swing-into-full-gear", "all-56-nomination-centres", "five-cornered-fight", "seven-police-reports"]),
    ("What issues did Pakatan Harapan candidates in Johor Baru focus on?", ["floods-causeway-jams", "vote-chase-intensifies"]),
    ("What was the turnout for early voting?", ["early-voter-turnout", "early-voting-concludes"]),
    ("Who are the candidates contesting Puteri Wangsa?", ["five-cornered-fight-shapes-up-in-puteri-wangsa", "maszlee-malik-casts-vote"]),
    ("Why did Pakatan Harapan perform poorly in the Johor election?", ["decline-in-malay-voter-support", "major-setback-for-pakatan", "pakatan-suffers-crushing-defeat"]),
    ("Who was sworn in as Johor Mentri Besar after the election?", ["sworn-in-as-johor-mentri-besar", "ahmad-zahid-congratulates"]),
    ("How many complaints did MCMC receive about online content during the campaign?", ["mcmc-received-29-complaints"]),
    ("How many police reports were lodged after nomination day?", ["seven-police-reports", "orderly-race-to-polling-day"]),
    ("What did Anwar Ibrahim say after the Johor election results?", ["despite-poll-outcome-says-anwar"]),
    ("What did Onn Hafiz say about state and federal government relations during the campaign?", ["state-federal-ties"]),
    ("How many candidates lost their election deposits?", ["barisan-reps-urged-to-stay-humble"]),
]
# Out-of-scope questions: the chatbot should decline these
OUT_OF_SCOPE = [
    "Who won the Melaka state election?",
    "What happened in the Negeri Sembilan state election?",
    "What is the latest Bursa Malaysia stock market index?",
    "Who won the latest Malaysia Cup football final?",
    "What is the weather forecast for Kuala Lumpur tomorrow?",
]
# Edge cases, reported but not scored: the PDF's Pulai example and a question about the previous (2022) election
EDGE = ["Who won the seat in Pulai?", "How many seats did Barisan Nasional win in the 2022 Johor election?"]


def main():
    chunks = load_chunks()
    questions = [q for q, _ in IN_SCOPE] + OUT_OF_SCOPE + EDGE
    kinds = ["in-scope"] * len(IN_SCOPE) + ["out-of-scope"] * len(OUT_OF_SCOPE) + ["edge"] * len(EDGE)
    expected = [keys for _, keys in IN_SCOPE] + [[]] * (len(OUT_OF_SCOPE) + len(EDGE))

    rows, summary = [], []
    for backend in config.EMBEDDING_FILES:
        try:
            passages = load_embeddings(backend, chunks)
        except RuntimeError as error:
            print(f"[{backend}] {error} Skipping.")
            continue
        question_vectors = embed(questions, backend, kind="question")

        try:                                                   # FAISS check: same results as NumPy?
            import faiss
            index = faiss.IndexFlatIP(passages.shape[1])
            index.add(np.ascontiguousarray(passages, dtype="float32"))
            faiss_scores, faiss_ids = index.search(np.ascontiguousarray(question_vectors, dtype="float32"), config.TOP_K)
        except ImportError:
            faiss_ids = None

        same = []
        for i, (question, kind, keys) in enumerate(zip(questions, kinds, expected)):
            top, scores = search(question_vectors[i], passages)
            if faiss_ids is not None:
                # same top-5 similarity scores (passages with tied scores may be listed in a different order)
                same.append(np.allclose(np.sort(scores), np.sort(faiss_scores[i]), atol=1e-4))
            urls = [chunks[j]["url"] for j in top]
            rank = next((r for r, url in enumerate(urls, 1) if any(k in url for k in keys)), None) if keys else None
            rows.append({"backend": backend, "kind": kind, "question": question,
                         "top_score": round(float(scores[0]), 3),
                         "correct_in_top5": (rank is not None) if keys else None,
                         "rank_of_correct": rank, "top_article": chunks[top[0]]["title"][:60]})
        faiss_same = all(same) if same else None

        df = pd.DataFrame([r for r in rows if r["backend"] == backend])
        ins, oos = df[df.kind == "in-scope"], df[df.kind == "out-of-scope"]
        lowest_in, highest_out = ins.top_score.min(), oos.top_score.max()
        gap = lowest_in - highest_out
        summary.append({
            "backend": backend,
            "hit_rate@5": round(ins.correct_in_top5.mean(), 3),
            "MRR@5": round(ins.rank_of_correct.apply(lambda r: 0 if pd.isna(r) else 1 / r).mean(), 3),
            "lowest in-scope score": lowest_in, "highest out-of-scope score": highest_out,
            "separation (gap)": round(gap, 3),
            "suggested threshold": round((lowest_in + highest_out) / 2, 3) if gap > 0 else "overlap - see table",
            "FAISS same as NumPy": faiss_same,
        })

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
    print("\nPer-question results")
    print(results.pivot_table(index=["kind", "question"], columns="backend",
                              values=["top_score", "rank_of_correct"], aggfunc="first").to_string())
    print("\nSummary")
    print(pd.DataFrame(summary).set_index("backend").T.to_string())

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.RESULTS_DIR / "retrieval_comparison.csv", index=False)
    pd.DataFrame(summary).to_csv(config.RESULTS_DIR / "retrieval_summary.csv", index=False)
    print(f"\nSaved details to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
