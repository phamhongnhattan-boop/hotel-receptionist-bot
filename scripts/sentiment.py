"""Hotel review sentiment: a cheap classic model vs Gemini, plus what guests praise or complain about.

    python -m scripts.sentiment                     # TF-IDF model + Gemini on 120 test reviews
    python -m scripts.sentiment --gemini-n 0        # TF-IDF only, no API calls
    python -m scripts.sentiment --gemini-n 200 --delay 5

Data: TripAdvisor Hotel Reviews (Alam et al., 2016; Kaggle "andrewmvd/trip-advisor-hotel-reviews",
CC BY-NC 4.0). Put tripadvisor_hotel_reviews.csv in data/ (data/ is not committed).
Labels from the stars: 1-2 negative, 3 neutral, 4-5 positive.

Questions this answers:
1. How good is a fast, free model (TF-IDF + logistic regression) on this task?
2. Is Gemini (zero-shot, no training) better, on the same test reviews? Is it worth the API calls?
3. Which parts of the stay (room, staff, breakfast...) do guests praise or complain about?
   Only Gemini gives this (aspect-based sentiment); a hotel manager can act on it.
Outputs (no review text is saved, only row numbers and labels, because of the data license):
  reports/sentiment_results.json, reports/sentiment_gemini_sample.csv, reports/review_aspects.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tripadvisor_hotel_reviews.csv"
REPORTS = ROOT / "reports"
LABELS = ["negative", "neutral", "positive"]
ASPECTS = ["room", "cleanliness", "staff", "breakfast_food", "location", "price_value", "noise", "facilities"]
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]
SEED = 42
logging.getLogger("httpx").setLevel(logging.WARNING)


def stars_to_label(stars: int) -> str:
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def load_reviews(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Review": "text", "Rating": "stars"})[["text", "stars"]].dropna()
    df = df[df["stars"].between(1, 5)].reset_index(drop=True)
    df["label"] = df["stars"].astype(int).map(stars_to_label)
    return df


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 3),
        "f1_per_class": dict(zip(LABELS, np.round(f1_score(y_true, y_pred, labels=LABELS, average=None,
                                                              zero_division=0), 3).tolist())),
        "confusion_rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def train_tfidf(texts, labels, cv: int = 3):
    """Word 1-2 grams + logistic regression. class_weight=balanced because neutral is rare (~11%)."""
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    search = GridSearchCV(pipe, {"logisticregression__C": [0.5, 1, 2, 4]}, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(texts, labels)
    return search.best_estimator_, search.best_params_["logisticregression__C"]


def top_words(model, k: int = 10) -> dict:
    """The words that push a review most toward each class: a quick check that the model is sensible."""
    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    names = np.array(vec.get_feature_names_out())
    return {c: names[np.argsort(-clf.coef_[i])[:k]].tolist() for i, c in enumerate(clf.classes_)}


# ---------- Gemini ----------
PROMPT = """Classify this hotel review.
sentiment: the guest's overall feeling (negative, neutral or positive).
aspects: only the parts of the stay the review really talks about, each with its polarity.
Allowed aspects: {aspects}.

REVIEW:
{review}"""


def gemini_schema():
    from google.genai import types
    return types.Schema(type="OBJECT", required=["sentiment", "aspects"], properties={
        "sentiment": types.Schema(type="STRING", enum=LABELS),
        "aspects": types.Schema(type="ARRAY", items=types.Schema(
            type="OBJECT", required=["aspect", "polarity"], properties={
                "aspect": types.Schema(type="STRING", enum=ASPECTS),
                "polarity": types.Schema(type="STRING", enum=["negative", "positive"]),
            })),
    })


def gemini_classify(client, review: str, max_chars: int = 4000) -> tuple[dict, str]:
    """One review -> {"sentiment", "aspects"}. Retries busy models, then tries the next one."""
    from google.genai import errors, types
    last = None
    for model in GEMINI_MODELS:
        for wait in (0, 10, 30):
            time.sleep(wait)
            try:
                resp = client.models.generate_content(
                    model=model, contents=PROMPT.format(aspects=", ".join(ASPECTS), review=review[:max_chars]),
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                       response_schema=gemini_schema()))
                return parse_gemini(resp.text), model
            except errors.APIError as e:
                if not (isinstance(e, errors.ServerError) or e.code == 429):
                    raise
                last = e
                print(f"    {model} busy ({e.code}), retrying...")
    raise last


def parse_gemini(text: str) -> dict:
    """Keep only allowed values, so a strange answer cannot break the counts."""
    data = json.loads(text)
    sentiment = data.get("sentiment") if data.get("sentiment") in LABELS else "neutral"
    aspects = {(a.get("aspect"), a.get("polarity")) for a in data.get("aspects", [])
               if a.get("aspect") in ASPECTS and a.get("polarity") in ("negative", "positive")}
    return {"sentiment": sentiment, "aspects": sorted(aspects)}


def aspect_summary(results: list[dict]) -> list[dict]:
    """Per aspect: how many reviews mention it, and the share that is negative."""
    count = Counter((a, p) for r in results for a, p in r["aspects"])
    out = []
    for a in ASPECTS:
        pos, neg = count[(a, "positive")], count[(a, "negative")]
        if pos + neg:
            out.append({"aspect": a, "mentions": pos + neg, "positive": pos, "negative": neg,
                        "negative_%": round(100 * neg / (pos + neg), 1)})
    return sorted(out, key=lambda r: -r["mentions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--gemini-n", type=int, default=120, help="test reviews sent to Gemini (0 = none)")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between Gemini calls")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    df = load_reviews(Path(args.data))
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"{len(df)} reviews | train {len(train)} | test {len(test)} | "
          f"labels {dict(Counter(df['label']))}")

    t0 = time.perf_counter()
    model, best_c = train_tfidf(train["text"], train["label"])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tfidf_pred = model.predict(test["text"])
    ms_per_review = (time.perf_counter() - t0) * 1000 / len(test)
    majority = Counter(train["label"]).most_common(1)[0][0]

    results = {
        "data": f"TripAdvisor Hotel Reviews, {len(df)} reviews, stratified 80/20 split (seed {SEED})",
        "labels": "1-2 stars negative, 3 neutral, 4-5 positive",
        "full_test_set": {
            "n": len(test),
            "majority_class": scores(test["label"], [majority] * len(test)),
            "tfidf_logreg": {**scores(test["label"], tfidf_pred), "C": best_c,
                             "train_seconds": round(train_s, 1), "ms_per_review": round(ms_per_review, 3)},
        },
        "tfidf_top_words": top_words(model),
    }
    print("TF-IDF + LogReg on the full test set:", json.dumps(results["full_test_set"]["tfidf_logreg"]))

    if args.gemini_n > 0:
        from app.llm import get_client
        client = get_client()
        # Same random sample for both models, so the comparison is fair
        sample = test.sample(n=min(args.gemini_n, len(test)), random_state=SEED)
        rows, gem, latencies, used = [], [], [], Counter()
        for i, (idx, r) in enumerate(sample.iterrows(), 1):
            t0 = time.perf_counter()
            out, model_name = gemini_classify(client, r["text"])
            latencies.append((time.perf_counter() - t0) * 1000)
            used[model_name] += 1
            gem.append(out)
            tf = model.predict([r["text"]])[0]
            rows.append({"row": int(idx), "stars": int(r["stars"]), "label": r["label"], "tfidf": tf,
                         "gemini": out["sentiment"], "gemini_model": model_name,
                         "aspects": ";".join(f"{a}:{p}" for a, p in out["aspects"])})
            print(f"[{i:03d}/{len(sample)}] true {r['label']:<8} tfidf {tf:<8} gemini {out['sentiment']:<8} "
                  f"{rows[-1]['aspects'][:70]}")
            time.sleep(args.delay)

        y = [r["label"] for r in rows]
        results["gemini_sample"] = {
            "n": len(rows),
            "models_used": dict(used),
            "tfidf_logreg": scores(y, [r["tfidf"] for r in rows]),
            "gemini_zero_shot": {**scores(y, [r["gemini"] for r in rows]),
                                 "ms_per_review_p50": round(float(np.median(latencies)))},
        }
        aspects = {"source": f"Gemini on {len(rows)} test reviews", "aspects": aspect_summary(gem)}
        with open(REPORTS / "review_aspects.json", "w", encoding="utf-8") as f:
            json.dump(aspects, f, indent=2)
        with open(REPORTS / "sentiment_gemini_sample.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\nOn the same", len(rows), "reviews:")
        for name in ("tfidf_logreg", "gemini_zero_shot"):
            s = results["gemini_sample"][name]
            print(f"  {name:<17} accuracy {s['accuracy']}  macro-F1 {s['macro_f1']}  F1 {s['f1_per_class']}")
        print("\nAspects (Gemini):")
        for a in aspects["aspects"]:
            print(f"  {a['aspect']:<15} {a['mentions']:>3} mentions, {a['negative_%']:>5}% negative")

    with open(REPORTS / "sentiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {REPORTS / 'sentiment_results.json'}")


if __name__ == "__main__":
    main()
