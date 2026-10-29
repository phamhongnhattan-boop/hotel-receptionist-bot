"""Quality check: send 30 test questions to the real bot and check the replies.

    python -m scripts.eval_questions              # all 30 questions (~3-4 minutes)
    python -m scripts.eval_questions --delay 10   # slower, if you hit rate limits (429)
    python -m scripts.eval_questions --rescore    # re-grade saved replies, no API calls

A reply PASSES if it contains ANY of the expected keywords (case-insensitive)
and NONE of the forbidden keywords (used to catch invented "we don't have X" answers).
Numbers are compared without separators, so "250.000", "250,000" and "250000" all match.
Results are saved to reports/eval_results.csv. Put the final pass rate in the README.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path


# Out-of-KB questions pass if the bot admits it does not know OR hands the guest to staff,
# and does NOT invent a "we don't have X" answer (checked by the forbidden lists below).
NOT_SURE_VI = ["không chắc", "chưa chắc", "chưa rõ", "không rõ", "không thấy thông tin",
               "chưa có thông tin", "không có thông tin", "không có sẵn", "hotline", "0236", "nhân viên", "lễ tân"]
NOT_SURE_EN = ["not sure", "don't have information", "do not have information", "no information",
               "hotline", "0236", "staff", "front desk"]
INVENTED_NO_VI = ["không có dịch vụ", "không có phòng", "không cung cấp"]
INVENTED_NO_EN = ["we don't have", "we do not have", "we don't offer", "we do not offer", "does not have", "doesn't have"]

# (category, question, expected keywords[, forbidden keywords])
CASES = [
    # --- Facts from the knowledge base (Vietnamese) ---
    ("facts_vi", "Mấy giờ được nhận phòng?", ["14:00", "14h", "2 giờ chiều"]),
    ("facts_vi", "Trả phòng trước mấy giờ?", ["12:00", "12h", "12 giờ"]),
    ("facts_vi", "Bữa sáng phục vụ từ mấy giờ?", ["6:30", "6h30"]),
    ("facts_vi", "Phòng Standard có bao gồm bữa sáng không?", ["150000"]),
    ("facts_vi", "Đưa đón sân bay bằng xe 7 chỗ giá bao nhiêu?", ["350000"]),
    ("facts_vi", "Hủy phòng có mất phí không?", ["48"]),
    ("facts_vi", "Trẻ em 4 tuổi có phải trả tiền không?", ["miễn phí", "không mất", "không phải trả"]),
    ("facts_vi", "Giường phụ giá bao nhiêu một đêm?", ["300000"]),
    ("facts_vi", "Gửi xe ô tô có mất phí không?", ["50000"]),
    ("facts_vi", "Khách sạn cách biển bao xa?", ["200 m", "200m", "200 mét"]),
    ("facts_vi", "Tôi có thể trả bằng đô la Mỹ không?", ["không chấp nhận", "không nhận", "chỉ nhận", "vnd"]),
    ("facts_vi", "Nhận phòng cần mang giấy tờ gì?", ["cccd", "hộ chiếu", "căn cước"]),
    # --- Facts from the knowledge base (English) ---
    ("facts_en", "What time is check-out?", ["12:00", "12 pm", "noon", "12pm"]),
    ("facts_en", "Do you have a swimming pool?", ["rooftop", "6:00", "6 am"]),
    ("facts_en", "Is the gym open at night?", ["24/7", "24 hours", "all day"]),
    ("facts_en", "Can I bring my dog?", ["not allow", "no pets", "do not accept", "don't accept", "not accept"]),
    ("facts_en", "Is Wi-Fi free?", ["free"]),
    ("facts_en", "How far is the airport?", ["6 km", "6km", "15 min"]),
    ("facts_en", "How far is Hoi An from the hotel?", ["30 km", "30km", "45 min"]),
    ("facts_en", "What is the fine for smoking in the room?", ["1000000"]),
    # --- Prices: the bot must call check_availability (no guessing) ---
    ("price_tool", "Giá phòng Standard 1 đêm từ 10/12 đến 11/12 là bao nhiêu?", ["900000"]),
    ("price_tool", "How much is a Deluxe room from December 5 to December 8?", ["4200000"]),
    ("price_tool", "Phòng Family 2 đêm từ 15/12 đến 17/12 hết bao nhiêu tiền?", ["4200000"]),
    ("price_tool", "Deluxe room price per night on December 20?", ["1400000"]),
    # --- Not in the knowledge base: the bot must NOT invent an answer ---
    ("out_of_kb", "Khách sạn có spa không?", NOT_SURE_VI, INVENTED_NO_VI),
    ("out_of_kb", "Có phòng karaoke không?", NOT_SURE_VI, INVENTED_NO_VI),
    ("out_of_kb", "What's on the dinner menu tonight?", NOT_SURE_EN, INVENTED_NO_EN),
    ("out_of_kb", "Do you offer a shuttle to Ba Na Hills?", NOT_SURE_EN, INVENTED_NO_EN),
    # --- Language: reply in the guest's language ---
    ("language", "Phòng Deluxe có ban công không?", ["ban công", "hướng biển"]),
    ("language", "Does the Family room have a bathtub?", ["bathtub", "yes"]),
]


def digits_only(text: str) -> str:
    return re.sub(r"(?<=\d)[.,\s](?=\d{3})", "", text)


def passes(reply: str, keywords: list[str], forbidden: list[str] | None = None) -> bool:
    low = reply.lower()
    if forbidden and any(f.lower() in low for f in forbidden):
        return False
    flat = digits_only(low)
    for k in keywords:
        k = k.lower()
        if k.isdigit():  # whole number only: "150000" must not match inside "1500000"
            if re.search(rf"(?<!\d){k}(?!\d)", flat):
                return True
        elif k in low:
            return True
    return False


def report(rows: list[dict]) -> None:
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += r["pass"]
        by_cat[r["category"]][1] += 1
    total = sum(p for p, _ in by_cat.values())
    print("\nBy category:")
    for cat, (p, n) in by_cat.items():
        print(f"  {cat:<11} {p}/{n}")
    print(f"\nPass rate: {total}/{len(rows)} = {total / len(rows):.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between questions (free tier rate limit)")
    parser.add_argument("--rescore", action="store_true", help="re-grade the saved replies without calling Gemini")
    args = parser.parse_args()
    out_path = Path(__file__).resolve().parents[1] / "reports" / "eval_results.csv"
    rules = {q: (kw, rest[0] if rest else None) for _, q, kw, *rest in CASES}

    if args.rescore:
        with open(out_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            keywords, forbidden = rules[r["question"]]
            r["pass"] = passes(r["reply"], keywords, forbidden)
            if not r["pass"]:
                print(f"FAIL ({r['category']}) {r['question']}\n        -> {r['reply'][:160]!r}")
    else:
        from app.main import build_agent  # imported here so --rescore works without API setup

        agent = build_agent()
        rows = []
        for i, (cat, question, keywords, *rest) in enumerate(CASES, 1):
            try:
                out = agent.reply(str(uuid.uuid4()), question)
                reply, model = out["reply"], out.get("model", "")
            except Exception as e:  # keep going, count as fail
                reply, model = f"ERROR: {e!r}", ""
            ok = passes(reply, keywords, rest[0] if rest else None)
            rows.append({"category": cat, "question": question, "pass": ok, "model": model,
                         "reply": reply.replace("\n", " ")})
            print(f"[{i:02d}/{len(CASES)}] {'PASS' if ok else 'FAIL'} ({cat}) {question}\n        -> {reply[:140]!r}")
            if i < len(CASES):
                time.sleep(args.delay)

    report(rows)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "question", "pass", "model", "reply"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
