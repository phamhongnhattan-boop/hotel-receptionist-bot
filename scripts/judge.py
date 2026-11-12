"""LLM-as-judge: a second Gemini model grades the saved replies of the 30-question test.

    python -m scripts.eval_questions      # first, to get reports/eval_results.csv
    python -m scripts.judge               # ~30 Gemini calls

Keyword grading (eval_questions.py) is strict but blind: "check-in from 14:00" and
"check-in is NOT from 14:00" both contain "14:00". The judge reads the hotel knowledge base,
the question and the reply, and answers 4 yes/no questions. Yes/no is easier to check and
more stable than a 1-10 score. The script also reports where the judge and the keyword
rules disagree, because a judge can be wrong too: those rows should be read by a person.
Results: reports/judge_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from google.genai import errors, types

from app import config
from app.llm import get_client

ROOT = Path(__file__).resolve().parents[1]
JUDGE_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # larger model first; lite only if it stays busy
logging.getLogger("httpx").setLevel(logging.WARNING)

RUBRIC = """You grade replies of a hotel chatbot. Use ONLY the hotel information below as the truth.

HOTEL INFORMATION:
{kb}

Room prices per night (from the booking system): standard 900000 VND, deluxe 1400000 VND, family 2100000 VND.

GUEST QUESTION: {question}
BOT REPLY: {reply}

Answer each question with true or false:
- faithful: every fact in the reply is supported by the hotel information or the room prices
  (saying "not sure, please call the hotline" is faithful).
- answers_question: the reply answers what the guest asked, or honestly says it does not know.
- right_language: the reply is in the same language as the question.
- no_invention: the reply does NOT claim the hotel has or does not have something that the
  hotel information does not mention.
Give a short reason (max 20 words) for any false."""

SCHEMA = types.Schema(type="OBJECT", properties={
    "faithful": types.Schema(type="BOOLEAN"),
    "answers_question": types.Schema(type="BOOLEAN"),
    "right_language": types.Schema(type="BOOLEAN"),
    "no_invention": types.Schema(type="BOOLEAN"),
    "reason": types.Schema(type="STRING"),
}, required=["faithful", "answers_question", "right_language", "no_invention", "reason"])
CRITERIA = ["faithful", "answers_question", "right_language", "no_invention"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=4.0)
    args = parser.parse_args()

    kb = "\n".join(p.read_text(encoding="utf-8") for p in sorted(config.KB_DIR.glob("*.md")))
    with open(ROOT / "reports" / "eval_results.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    client = get_client()

    def judge(prompt: str) -> tuple[dict, str]:
        """Try each model a few times with growing waits: 503 "high demand" is usually short."""
        last = None
        for model in JUDGE_MODELS:
            for wait in (0, 10, 30):
                time.sleep(wait)
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                           response_schema=SCHEMA))
                    return json.loads(resp.text), model
                except errors.APIError as e:
                    if not (isinstance(e, errors.ServerError) or e.code == 429):
                        raise
                    last = e
                    print(f"        {model} busy ({e.code}), retrying...")
        raise last

    out = []
    for i, r in enumerate(rows, 1):
        verdict, judge_model = judge(RUBRIC.format(kb=kb, question=r["question"], reply=r["reply"]))
        ok = all(verdict[c] for c in CRITERIA)
        keyword_ok = r["pass"] == "True"
        out.append({"category": r["category"], "question": r["question"], "keyword_pass": keyword_ok,
                    "judge_pass": ok, **{c: verdict[c] for c in CRITERIA}, "reason": verdict["reason"],
                    "judge_model": judge_model, "reply": r["reply"]})
        flag = "" if ok == keyword_ok else "   <- judge and keywords disagree"
        print(f"[{i:02d}/{len(rows)}] judge {'PASS' if ok else 'FAIL'} | keywords {'PASS' if keyword_ok else 'FAIL'}"
              f" | {r['question'][:60]}{flag}")
        if not ok:
            print(f"        reason: {verdict['reason']}")
        time.sleep(args.delay)

    n = len(out)
    print("\nJudge results:")
    for c in CRITERIA:
        print(f"  {c:<17} {sum(o[c] for o in out)}/{n}")
    print(f"  all four          {sum(o['judge_pass'] for o in out)}/{n}")
    agree = sum(o["judge_pass"] == o["keyword_pass"] for o in out)
    print(f"Judge agrees with keyword grading on {agree}/{n} replies")

    path = ROOT / "reports" / "judge_results.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
