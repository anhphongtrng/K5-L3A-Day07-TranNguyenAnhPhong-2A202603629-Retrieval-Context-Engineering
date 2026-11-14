
import csv
import json
import time
from pathlib import Path

from retrieval import Retriever
from agent import (
    answer_question,
    build_context,
    build_context_improved,
    validate_answer,
    ABSTAIN,
)

DATA_PATH = Path(
    "../data/benchmark/benchmark_calculus_v1_24q.json"
)
OUT_DIR = Path("../results")
TOP_K = 3


def main():
    benchmark = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    questions = [
        q for q in benchmark["questions"]
        if q["split"] == "dev"
    ]

    retriever = Retriever()
    rows = []
    traces = []

    try:
        for q in questions:
            question = q["question"]

            # No gold information is passed to retrieval.
            section = None

            # Section must come from the question itself.
            from agent import section_from_question
            section = section_from_question(question)

            retrieval_start = time.perf_counter()

            hits = retriever.search(
                "hybrid",
                question,
                top_k=TOP_K,
                section=section
            )

            retrieval_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

            for mode in ("baseline", "improved"):
                if mode == "baseline":
                    context, selected = build_context(hits)
                else:
                    context, selected = build_context_improved(
                        hits, question
                    )

                answer_result = answer_question(
                    question, context
                )

                raw_answer = answer_result["content"]
                valid, citations, validation = (
                    validate_answer(raw_answer, selected)
                )

                final_answer = (
                    raw_answer if valid else ABSTAIN
                )

                usage = answer_result.get("usage", {})

                rows.append({
                    "question_id": q["id"],
                    "mode": mode,
                    "retrieval_method": "hybrid",
                    "top_k": TOP_K,
                    "context_chars": len(context),
                    "selected_chunks": len(selected),
                    "answer": final_answer,
                    "abstained": final_answer == ABSTAIN,
                    "citation_validation": validation,
                    "total_tokens": usage.get(
                        "total_tokens", 0
                    ),
                    "api_cost_usd": usage.get(
                        "cost", 0
                    ),
                    "retrieval_ms": round(
                        retrieval_ms, 2
                    ),
                    "answer_ms": answer_result.get(
                        "latency_ms", 0
                    ),
                })

                traces.append({
                    "question_id": q["id"],
                    "mode": mode,
                    "question": question,
                    "retrieval": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "page": chunk.page,
                            "source": chunk.source,
                            "section": chunk.section,
                            "score": score,
                            "content": chunk.content,
                        }
                        for chunk, score in hits
                    ],
                    "context": context,
                    "selected_chunks": selected,
                    "raw_answer": raw_answer,
                    "answer": final_answer,
                    "citations": citations,
                    "citation_validation": validation,
                    "usage": usage,
                })

                print(
                    q["id"], mode,
                    "tokens:", usage.get(
                        "total_tokens", 0
                    ),
                    "abstained:", final_answer == ABSTAIN
                )

    finally:
        retriever.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "context_dev.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    trace_path = OUT_DIR / "context_dev_trace.json"
    trace_path.write_text(
        json.dumps(
            traces,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== CONTEXT BENCHMARK ===")
    print("Questions:", len(questions))
    print("Runs:", len(rows))
    print("CSV:", csv_path)
    print("Trace:", trace_path)

    for mode in ("baseline", "improved"):
        group = [r for r in rows if r["mode"] == mode]

        print(
            mode,
            "Avg chars:",
            round(
                sum(r["context_chars"] for r in group)
                / len(group), 1
            ),
            "Total tokens:",
            sum(
                int(r["total_tokens"] or 0)
                for r in group
            ),
            "Total API cost:",
            round(
                sum(
                    float(r["api_cost_usd"] or 0)
                    for r in group
                ), 6
            )
        )


if __name__ == "__main__":
    main()
