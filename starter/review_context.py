
import json
import re
from pathlib import Path

BENCHMARK = Path("../data/benchmark/benchmark_calculus_v1_24q.json")
TRACE = Path("../results/context_dev_trace.json")
OUTPUT = Path("../results/context_dev_review.txt")


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
traces = json.loads(TRACE.read_text(encoding="utf-8"))

questions = {
    q["id"]: q
    for q in benchmark["questions"]
    if q["split"] == "dev"
}

lines = []
suspected_missing = []

for item in traces:
    q = questions[item["question_id"]]
    evidence = q.get("evidence_quote") or ""
    context = item["context"]

    exact_match = (
        normalize(evidence) in normalize(context)
        if evidence else None
    )

    if q.get("answerable", True) and not exact_match:
        suspected_missing.append(
            (item["question_id"], item["mode"])
        )

    lines.extend([
        "=" * 70,
        f"QUESTION: {item['question_id']}",
        f"MODE: {item['mode']}",
        f"QUESTION TEXT: {q['question']}",
        f"EXPECTED ANSWER: {q.get('expected_answer')}",
        f"GOLD EVIDENCE: {evidence}",
        f"EXACT EVIDENCE FOUND: {exact_match}",
        f"ANSWER: {item['answer']}",
        f"CITATIONS: {item['citations']}",
        "",
        "CONTEXT:",
        context,
        "",
        "CONTEXT SCORE (0/1/2):",
        "ANSWER SCORE (0/1/2):",
        "CITATION SUPPORTED (yes/no):",
        "NOTES:",
        "",
    ])

OUTPUT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("Total runs:", len(traces))
print("Review file:", OUTPUT)
print("Potential evidence mismatches:", len(suspected_missing))

for qid, mode in suspected_missing:
    print(qid, mode)
