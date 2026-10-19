"""
Step 3b - End-to-end DEVELOPMENT checks: does the chatbot answer in-scope questions with valid citations,
and decline everything else? These questions were also used while choosing the settings (e.g. the cut-off),
so they are not an independent test; final_check.py holds fresh questions run once after the settings were frozen.

"keyword_match" is a quick diagnostic (does the answer contain e.g. "48"?), not proof of correctness:
read the answers in the CSV against their cited articles.

Run:  python evaluate.py      (about 2-3 minutes; it pauses between questions for Gemini's free-tier limits)
Results are saved to results/answer_evaluation.csv.
"""
import time

import pandas as pd

import config
from rag import answer

# (question, type, keyword(s) a correct answer should contain - any of the listed forms)
TESTS = [
    ("How many seats did Barisan Nasional win in the 2026 Johor state election?", "in-scope", ["48"]),
    ("When was nomination day for the Johor polls?", "in-scope", ["27"]),
    ("What was the turnout for early voting?", "in-scope", ["94.49"]),
    ("Who was sworn in as Johor Mentri Besar after the election?", "in-scope", ["Onn Hafiz"]),
    ("How many complaints did MCMC receive during the campaign?", "in-scope", ["29"]),
    ("How many candidates lost their election deposits?", "in-scope", ["55"]),
    ("What issues did Pakatan Harapan candidates in Johor Baru campaign on?", "in-scope", ["flood", "Causeway"]),
    ("Why did Pakatan Harapan lose in Johor?", "in-scope", ["Malay"]),
    ("Who won the Melaka state election?", "out-of-scope", []),
    ("What happened in the Negeri Sembilan state election?", "out-of-scope", []),
    ("What is the latest Bursa Malaysia stock market index?", "out-of-scope", []),
    ("Who won the latest Malaysia Cup football final?", "out-of-scope", []),
    ("Who should I vote for in the next election?", "opinion", []),
    ("Who won the seat in Pulai?", "edge", []),
]


def main():
    rows = []
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for number, (question, kind, facts) in enumerate(TESTS, start=1):
        try:
            result = answer(question)
        except Exception as error:                      # record the failure and carry on
            result = {"answer": f"ERROR: {str(error)[:200]}", "sources": [], "declined": True,
                      "reason": "error", "top_score": None, "model": None}
        error = result["reason"] == "error"
        keyword_match = any(f.lower() in result["answer"].lower() for f in facts) if facts and not result["declined"] else None
        if error:                                       # a service failure is never "correct behaviour"
            correct_behaviour = False
        elif kind == "in-scope":
            correct_behaviour = not result["declined"] and bool(result["sources"])
        else:
            correct_behaviour = result["declined"]
        rows.append({"question": question, "type": kind, "declined": result["declined"], "reason": result["reason"],
                     "top_score": result["top_score"], "model": result.get("model"), "cited_articles": len(result["sources"]),
                     "service_error": error, "keyword_match": keyword_match, "correct_behaviour": correct_behaviour,
                     "answer": result["answer"],
                     "sources": " | ".join(f"{s['title']} ({s['date']})" for s in result["sources"])})
        print(f"\n{number}. [{kind}] {question}")
        print(f"   -> {'DECLINED' if result['declined'] else 'ANSWERED'} ({result['reason']}, top score {result['top_score']}, model {result.get('model')})")
        print("   " + result["answer"].replace("\n", " ")[:400])
        for s in result["sources"]:
            print(f"      [{s['number']}] {s['title']} ({s['date']})")
        pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "answer_evaluation.csv", index=False)   # save as we go
        time.sleep(4)                                   # stay well inside the free-tier request limits

    df = pd.DataFrame(rows)
    in_scope, other = df[df.type == "in-scope"], df[df.type != "in-scope"]
    print()
    print("=== Summary (development checks) ===")
    print(f"In-scope answered with valid citations: {in_scope.correct_behaviour.sum()}/{len(in_scope)}")
    print(f"In-scope answers with keyword match (quick check, not proof): {in_scope.keyword_match.fillna(False).sum()}/{len(in_scope)}")
    print(f"Out-of-scope / opinion / edge declined: {other.correct_behaviour.sum()}/{len(other)}")
    print(f"Errors: {(df.reason == 'error').sum()}  |  Answer models used: {sorted(df.model.dropna().unique())}")
    df.to_csv(config.RESULTS_DIR / "answer_evaluation.csv", index=False)
    print(f"Saved to {config.RESULTS_DIR / 'answer_evaluation.csv'}  (read the answers to check they match their sources)")


if __name__ == "__main__":
    main()
