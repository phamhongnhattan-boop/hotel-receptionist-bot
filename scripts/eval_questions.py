"""Quick quality check: send test questions to the real bot and check keywords in the reply.

    python -m scripts.eval_questions

Add more cases over time (aim for 30-50). Put the final pass rate in the README.
"""
from __future__ import annotations

import uuid

from app.main import build_agent

# (question, keywords - reply passes if it contains ANY of them, case-insensitive)
CASES = [
    ("Mấy giờ được nhận phòng?", ["14:00", "14h", "2 giờ chiều"]),
    ("What time is check-out?", ["12:00", "12 pm", "noon"]),
    ("Bữa sáng có bao gồm trong phòng Standard không?", ["150.000", "150,000", "không bao gồm"]),
    ("Do you have a swimming pool?", ["pool", "rooftop"]),
    ("Đưa đón sân bay bao nhiêu tiền?", ["250.000", "250,000"]),
    ("Can I bring my dog?", ["not allow", "no pets", "do not accept", "don't accept"]),
    ("Hủy phòng có mất phí không?", ["48"]),
    ("Khách sạn có spa không?", ["không chắc", "hotline", "0236", "nhân viên"]),  # not in KB
]


def main():
    agent = build_agent()
    passed = 0
    for question, keywords in CASES:
        reply = agent.reply(str(uuid.uuid4()), question)["reply"]
        ok = any(k.lower() in reply.lower() for k in keywords)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question}\n    -> {reply[:160]!r}\n")
    print(f"Pass rate: {passed}/{len(CASES)} = {passed / len(CASES):.0%}")


if __name__ == "__main__":
    main()
