"""LAB7 held-out TEST: frozen Hybrid RRF top-3 + C1 context.

Runs the Answer LLM only (no LLM planner); the method is frozen to
Hybrid for a controlled evaluation. Gold labels are NEVER passed to
retrieval or the Answer LLM. Results persist after every question,
so interruption does not discard paid API calls. Re-running skips
completed cases unless --rerun is supplied.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import (
    ABSTAIN,
    CONTEXT_BUDGET_CHARS,
    TOP_K,
    answer_question,
    build_context,
    section_from_question,
    validate_answer,
)
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT / "data/benchmark/benchmark_calculus_v1_24q.json"
OUT_DIR = ROOT / "results"
CSV_PATH = OUT_DIR / "agent_test.csv"
TRACE_PATH = OUT_DIR / "agent_test_trace.json"

METHOD = "hybrid"
CONTEXT_MODE = "baseline"
CSV_FIELDS = [
    "id", "split", "retrieval_method", "context_mode", "top_k",
    "context_budget_chars", "filter_section", "num_retrieved",
    "num_selected", "context_chars", "answer", "abstained",
    "citations", "citation_validation", "retrieval_ms", "answer_ms",
    "total_latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
    "api_cost_usd", "model", "error",
]


def load_test_questions():
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    questions = [q for q in data["questions"] if q["split"].lower() == "test"]
    if len(questions) != 7:
        raise ValueError(f"Expected 7 TEST questions, found {len(questions)}")
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate TEST question IDs")
    # Preserve benchmark ordering; no answer labels are sent to the agent.
    return questions


def write_outputs(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_trace = TRACE_PATH.with_suffix(".json.tmp")
    temporary_trace.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary_trace.replace(TRACE_PATH)

    rows = []
    for record in records:
        usage = record.get("usage") or {}
        rows.append({
            "id": record["id"],
            "split": "test",
            "retrieval_method": METHOD,
            "context_mode": CONTEXT_MODE,
            "top_k": TOP_K,
            "context_budget_chars": CONTEXT_BUDGET_CHARS,
            "filter_section": record.get("filter_section") or "",
            "num_retrieved": len(record.get("retrieved") or []),
            "num_selected": len(record.get("selected_chunks") or []),
            "context_chars": len(record.get("context") or ""),
            "answer": record.get("answer") or "",
            "abstained": record.get("abstained", ""),
            "citations": "|".join(record.get("citations") or []),
            "citation_validation": record.get("citation_validation") or "",
            "retrieval_ms": record.get("retrieval_ms", ""),
            "answer_ms": record.get("answer_ms", ""),
            "total_latency_ms": record.get("total_latency_ms", ""),
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
            "api_cost_usd": usage.get("cost", ""),
            "model": record.get("model") or "",
            "error": record.get("error") or "",
        })

    temporary_csv = CSV_PATH.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(CSV_PATH)


def run_case(question_record, retriever, model):
    question = question_record["question"]
    section = section_from_question(question)
    record = {
        "id": question_record["id"],
        "question": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "retrieval_method": METHOD,
            "context_mode": CONTEXT_MODE,
            "strategy": "B",
            "top_k": TOP_K,
            "context_budget_chars": CONTEXT_BUDGET_CHARS,
            "model": model,
            "llm_planner_used": False,
            "max_retrieval_calls": 1,
        },
        "filter_section": section,
        "retrieved": [],
        "selected_chunks": [],
        "context": "",
        "answer": None,
        "raw_answer": None,
        "citations": [],
        "abstained": None,
        "citation_validation": None,
        "usage": {},
        "error": None,
    }
    start = time.perf_counter()
    try:
        search_start = time.perf_counter()
        hits = retriever.search(METHOD, question, top_k=TOP_K, section=section)
        record["retrieval_ms"] = round((time.perf_counter() - search_start) * 1000, 2)
        record["retrieved"] = [
            {
                "rank": i,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "source": chunk.source,
                "section": chunk.section,
                "score": float(score),
                "content": chunk.content,
            }
            for i, (chunk, score) in enumerate(hits, start=1)
        ]

        context, selected = build_context(hits, budget_chars=CONTEXT_BUDGET_CHARS)
        record["context"] = context
        record["selected_chunks"] = selected

        response = answer_question(question, context)
        raw = response["content"]
        valid, citations, validation = validate_answer(raw, selected)
        record["raw_answer"] = raw
        record["answer"] = raw if valid else ABSTAIN
        record["citations"] = citations if validation == "valid_ids" else []
        record["citation_validation"] = validation
        record["abstained"] = record["answer"] == ABSTAIN
        record["usage"] = response.get("usage") or {}
        record["model"] = response.get("model") or model
        record["answer_ms"] = response.get("latency_ms", 0)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Repeat all TEST calls and overwrite previous results")
    args = parser.parse_args()

    model = os.environ.get("OPENROUTER_MODEL")
    if not os.environ.get("OPENROUTER_API_KEY") or not model:
        raise SystemExit("Set OPENROUTER_API_KEY and OPENROUTER_MODEL first")
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("Set DATABASE_URL first")

    questions = load_test_questions()
    records = []
    if TRACE_PATH.exists() and not args.rerun:
        records = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        previous_models = {
            r.get("config", {}).get("model") for r in records
        }
        if previous_models and previous_models != {model}:
            raise SystemExit("Model differs from existing TEST trace. Do not mix models; use the frozen model.")
    done = {r["id"] for r in records if r.get("error") is None and r.get("answer") is not None}

    print(f"Frozen TEST configuration: {METHOD}, top-{TOP_K}, {CONTEXT_MODE}, {CONTEXT_BUDGET_CHARS} chars")
    print(f"Model: {model}; LLM Planner: disabled for controlled TEST")
    print(f"TEST questions: {len(questions)}; already completed: {len(done)}")

    retriever = Retriever()
    try:
        for q in questions:
            if q["id"] in done:
                print(f"{q['id']}: already completed, skip")
                continue
            result = run_case(q, retriever, model)
            records = [r for r in records if r["id"] != q["id"]]
            records.append(result)
            records.sort(key=lambda r: [x["id"] for x in questions].index(r["id"]))
            write_outputs(records)
            usage = result["usage"]
            print(
                f"{q['id']}: abstained={result['abstained']} "
                f"citations={len(result['citations'])} "
                f"tokens={usage.get('total_tokens', 'N/A')} "
                f"cost_usd={usage.get('cost', 'N/A')} "
                f"error={result['error']}"
            )
    finally:
        retriever.close()

    success = [r for r in records if not r.get("error") and r.get("answer") is not None]
    print("\n=== AGENT TEST SUMMARY ===")
    print("Completed:", len(success), "/", len(questions))
    print("Total answer tokens:", sum(int(r["usage"].get("total_tokens") or 0) for r in success))
    print("Total Answer API cost (USD):", round(sum(float(r["usage"].get("cost") or 0) for r in success), 8))
    print("Mean total latency (ms):", round(sum(r["total_latency_ms"] for r in success) / len(success), 2) if success else "N/A")
    print("CSV:", CSV_PATH)
    print("Trace:", TRACE_PATH)
    print("Note: Answer-only evaluation; excludes LLM planner costs. Gold labels never sent to model.")


if __name__ == "__main__":
    main()
