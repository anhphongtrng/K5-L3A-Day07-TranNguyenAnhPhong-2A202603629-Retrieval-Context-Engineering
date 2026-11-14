import argparse
import csv
import json
import re
import time
from pathlib import Path

from retrieval import Retriever


DATA_PATH = Path(
    "../data/benchmark/benchmark_calculus_v1_24q.json"
)

OUT_DIR = Path("../results")


def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9 ]+",
            " ",
            (text or "").lower()
        )
    ).strip()


def evidence_found(chunk, question):
    """
    Automated evidence proxy.
    Only the evaluator uses gold labels.
    """

    gold_page = question.get("gold_page")
    quote = normalize(question.get("evidence_quote"))

    if (
        not gold_page
        or not quote
        or int(chunk.page) != int(gold_page)
    ):
        return False

    candidate = normalize(chunk.content)
    words = quote.split()

    if len(words) <= 3:
        return quote in candidate

    overlap = (
        sum(w in candidate.split() for w in set(words))
        / len(set(words))
    )

    return quote in candidate or overlap >= 0.85


def pack_context(results, budget_chars=3000):
    """Assemble context under a character budget."""

    parts = []
    selected = []
    total = 0

    for chunk, score in results:
        text = (
            f"[source={chunk.source}; "
            f"page={chunk.page}; "
            f"chunk={chunk.chunk_id}]\n"
            f"{chunk.content}"
        )

        if total + len(text) > budget_chars:
            continue

        parts.append(text)
        selected.append(chunk)
        total += len(text)

    return "\n\n".join(parts), selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev"
    )

    parser.add_argument(
        "--budget-chars",
        type=int,
        default=3000
    )

    args = parser.parse_args()

    data = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    questions = [
        q for q in data["questions"]
        if q["split"] == args.split
    ]

    OUT_DIR.mkdir(exist_ok=True)

    retriever = Retriever()

    print("Strategy B chunks:", len(retriever.chunks))

    sections = {
        c.section
        for c in retriever.chunks
        if c.section is not None
    }

    print("Available section labels:", sorted(sections))

    # Verify that requested section filters exist.
    missing_sections = sorted(
        {
            str(q["filter"]["section"])
            for q in questions
            if q.get("filter")
            and q["filter"].get("section") is not None
        } - sections
    )

    if missing_sections:
        raise RuntimeError(
            f"Missing metadata sections: {missing_sections}. "
            "Check LAB6 section metadata."
        )

    rows = []

    for method in ("dense", "bm25", "hybrid"):
        for question in questions:

            # A benchmark-specified filter must be
            # justified by the question itself, not gold.
            section = (
                question.get("filter") or {}
            ).get("section")

            start = time.perf_counter()

            hits = retriever.search(
                method,
                question["question"],
                top_k=3,
                section=section
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            answerable = (
                question.get("answerable") is not False
                and question.get("expected_answer") is not None
            )

            flags = [
                evidence_found(chunk, question)
                if answerable else False
                for chunk, _ in hits
            ]

            context, packed = pack_context(
                hits,
                budget_chars=args.budget_chars
            )

            context_hit = (
                any(
                    evidence_found(chunk, question)
                    for chunk in packed
                )
                if answerable else None
            )

            rows.append({
                "method": method,
                "split": args.split,
                "id": question["id"],
                "answerable": answerable,
                "filter_section": section or "",
                "top1_evidence_proxy": (
                    int(bool(flags and flags[0]))
                    if answerable else ""
                ),
                "top3_evidence_proxy": (
                    int(any(flags))
                    if answerable else ""
                ),
                "context_evidence_proxy": (
                    int(context_hit)
                    if answerable else ""
                ),
                "latency_ms": round(elapsed_ms, 2),
                "context_chars": len(context),
                "context_tokens_estimate": round(
                    len(context.split()) * 1.33
                ),
                "ranks_chunk_ids": "|".join(
                    c.chunk_id for c, _ in hits
                ),
                "ranks_pages": "|".join(
                    str(c.page) for c, _ in hits
                ),
                "ranks_scores": "|".join(
                    f"{score:.4f}" for _, score in hits
                ),
                "context_chunk_ids": "|".join(
                    c.chunk_id for c in packed
                )
            })

    output_path = (
        OUT_DIR / f"retrieval_{args.split}.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults:", output_path)
    print(
        "Metrics are automated proxies. "
        "Review evidence manually before reporting."
    )

    for method in ("dense", "bm25", "hybrid"):

        group = [
            r for r in rows
            if r["method"] == method
        ]

        valid = [
            r for r in group
            if r["answerable"]
        ]

        n = len(valid)

        hits1 = sum(
            r["top1_evidence_proxy"]
            for r in valid
        )

        hits3 = sum(
            r["top3_evidence_proxy"]
            for r in valid
        )

        context_hits = sum(
            r["context_evidence_proxy"]
            for r in valid
        )

        avg_ms = (
            sum(r["latency_ms"] for r in group)
            / len(group)
        )

        print(
            f"{method:6} | "
            f"Hit@1 {hits1}/{n} | "
            f"Evidence Hit@3 {hits3}/{n} | "
            f"Context evidence {context_hits}/{n} | "
            f"Mean latency {avg_ms:.1f} ms"
        )

    retriever.close()


if __name__ == "__main__":
    main()
