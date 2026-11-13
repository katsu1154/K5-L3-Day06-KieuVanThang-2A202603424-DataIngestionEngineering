"""Run the frozen dev/test retrieval benchmark and prepare human audit sheets."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import chromadb
from dotenv import load_dotenv

from .embeddings import ChromaLocalEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_corpus_units
from .models import DocumentPage
from .store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STRATEGIES = ("A", "B", "C")
TOP_K = 3
COHERENCE_SAMPLE_SIZE = 10
METADATA_FIELDS = ("document_id", "source", "page", "unit_id", "volume")


def _configured_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    if not value.strip():
        raise ValueError("CHROMA_PERSIST_DIRECTORY must not be empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as question_file:
        for line_number, line in enumerate(question_file, start=1):
            if not line.strip():
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            if not isinstance(question, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
            required = (
                "id",
                "split",
                "question",
                "expected_answer",
                "gold_source_id",
                "gold_page",
                "evidence_quote",
            )
            missing = [key for key in required if key not in question]
            if missing:
                raise ValueError(
                    f"Missing fields on line {line_number} of {path}: {', '.join(missing)}"
                )
            question_id = str(question["id"])
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question id {question_id!r} in {path}")
            seen_ids.add(question_id)
            if question["split"] not in ("dev", "test"):
                raise ValueError(f"Invalid split for question {question_id!r}")
            if not all(
                isinstance(question[key], str) and question[key].strip()
                for key in ("question", "expected_answer", "evidence_quote")
            ):
                raise ValueError(f"Question {question_id!r} has empty question, answer, or evidence")
            if question.get("filter") is not None and not isinstance(question["filter"], dict):
                raise ValueError(f"Filter for question {question_id!r} must be an object or null")
            questions.append(question)

    dev_count = sum(question["split"] == "dev" for question in questions)
    test_count = sum(question["split"] == "test" for question in questions)
    filter_questions = [question for question in questions if question.get("filter")]
    test_filter_questions = [
        question for question in filter_questions if question["split"] == "test"
    ]
    if len(questions) < 20 or dev_count < 15 or test_count < 5:
        raise ValueError(
            f"Benchmark needs at least 20 questions (15 dev / 5 test); "
            f"found {len(questions)} ({dev_count} / {test_count})"
        )
    if len(filter_questions) < 3 or not test_filter_questions:
        raise ValueError("Benchmark needs at least 3 filter questions, including a test question")
    return questions


def _validate_questions(
    questions: list[dict[str, Any]],
    units: dict[str, DocumentPage],
) -> None:
    for question in questions:
        gold_unit_id = str(question.get("gold_unit_id") or question["gold_page"])
        if gold_unit_id not in units:
            raise ValueError(
                f"Question {question['id']!r} references unknown unit {gold_unit_id!r}"
            )
        gold_page = units[gold_unit_id]
        if gold_page.document_id != question["gold_source_id"]:
            raise ValueError(
                f"Question {question['id']!r} gold source does not match unit {gold_unit_id!r}"
            )
        if question.get("gold_logical_page") is not None:
            if int(question["gold_logical_page"]) != gold_page.page:
                raise ValueError(
                    f"Question {question['id']!r} gold page does not match unit {gold_unit_id!r}"
                )
        if question["evidence_quote"] not in gold_page.text:
            raise ValueError(
                f"Question {question['id']!r} evidence quote is not present in its gold unit"
            )
        for key, value in (question.get("filter") or {}).items():
            if gold_page.metadata.get(key) != value:
                raise ValueError(
                    f"Question {question['id']!r} filter {key}={value!r} "
                    "does not match its gold source metadata"
                )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _preserve_existing_manual_fields(
    path: Path,
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    manual_fields: tuple[str, ...],
) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_fields = set(key_fields + manual_fields)
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"Cannot preserve manual labels in {path}; missing columns: "
                f"{', '.join(sorted(missing_fields))}"
            )
        existing: dict[tuple[str, ...], dict[str, str]] = {}
        for old_row in reader:
            key = tuple(old_row[field] for field in key_fields)
            existing[key] = {field: old_row[field] for field in manual_fields}

    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        old_fields = existing.get(key)
        if old_fields is None:
            continue
        for field in manual_fields:
            if row[field] in ("", None) and old_fields[field]:
                row[field] = old_fields[field]


def _paragraph_sample(page: DocumentPage) -> tuple[int, int, str]:
    offset = 0
    for paragraph in page.text.split("\n\n"):
        start = offset
        end = start + len(paragraph)
        if len(paragraph.strip()) >= 80:
            return start, end, paragraph
        offset = end + 2
    raise ValueError(f"No paragraph of at least 80 characters in unit {page.metadata['unit_id']}")


def _create_coherence_audit(
    store: ChromaVectorStore,
    pages: list[DocumentPage],
    output_path: Path,
) -> None:
    samples = sorted(pages, key=lambda page: page.page)[:COHERENCE_SAMPLE_SIZE]
    if len(samples) != COHERENCE_SAMPLE_SIZE:
        raise ValueError(
            f"Coherence audit requires {COHERENCE_SAMPLE_SIZE} source units; found {len(samples)}"
        )

    rows: list[dict[str, Any]] = []
    for sample_index, page in enumerate(samples, start=1):
        source_start, source_end, source_paragraph = _paragraph_sample(page)
        target_offset = source_start + (source_end - source_start) // 2
        unit_id = str(page.metadata["unit_id"])
        for strategy in STRATEGIES:
            candidates: list[tuple[int, str, dict[str, Any]]] = []
            for stored_chunk in store.get_chunks(strategy, {"unit_id": unit_id}):
                chunk_id = stored_chunk["chunk_id"]
                metadata = stored_chunk["metadata"]
                start = int(metadata.get("chunk_start_char", -1))
                end = int(metadata.get("chunk_end_char", -1))
                if start <= target_offset < end:
                    candidates.append((start, chunk_id, metadata | {"text": stored_chunk["text"]}))
            if not candidates:
                raise RuntimeError(
                    f"Strategy {strategy} has no chunk covering the selected location "
                    f"in {unit_id}"
                )
            _, chunk_id, selected = min(candidates, key=lambda candidate: candidate[0])
            rows.append(
                {
                    "sample_id": f"coh-{sample_index:02d}",
                    "strategy": strategy,
                    "source_id": page.document_id,
                    "unit_id": unit_id,
                    "logical_page": page.page,
                    "selected_location_rule": (
                        "First paragraph >=80 characters in each of the 10 lowest logical_page units; "
                        "target is the paragraph midpoint."
                    ),
                    "source_paragraph_start_char": source_start,
                    "source_paragraph_end_char": source_end,
                    "target_offset_char": target_offset,
                    "source_paragraph": source_paragraph,
                    "chunk_id": chunk_id,
                    "chunk_start_char": selected["chunk_start_char"],
                    "chunk_end_char": selected["chunk_end_char"],
                    "chunk_text": selected["text"],
                    "manual_score_0_to_2": "",
                    "manual_reason": "",
                }
            )
    _preserve_existing_manual_fields(
        output_path,
        rows,
        ("sample_id", "strategy", "chunk_id"),
        ("manual_score_0_to_2", "manual_reason"),
    )
    _write_csv(output_path, list(rows[0]), rows)


def _create_human_audits(
    traces: list[dict[str, Any]],
    pages: list[DocumentPage],
    store: ChromaVectorStore,
    results_directory: Path,
) -> dict[str, Any]:
    unit_truth = {
        str(page.metadata["unit_id"]): {
            "document_id": page.document_id,
            "source": page.source,
            "page": page.page,
            "volume": page.metadata.get("volume"),
        }
        for page in pages
    }

    test_traces = [trace for trace in traces if trace["split"] == "test"]
    metadata_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    for trace in test_traces:
        for hit in trace["hits"][:TOP_K]:
            metadata = hit.get("metadata") or {}
            unit_id = str(metadata.get("unit_id", ""))
            truth = unit_truth.get(unit_id, {})
            compared_fields = {
                "document_id": truth.get("document_id"),
                "source": truth.get("source"),
                "page": truth.get("page"),
                "unit_id": unit_id if truth else None,
                "volume": truth.get("volume"),
            }
            field_results = {
                field: metadata.get(field) == expected
                for field, expected in compared_fields.items()
            }
            requested_filter = trace.get("filter") if trace["strategy"] == "C" else None
            filter_compliant: bool | None = None
            if requested_filter:
                filter_compliant = all(
                    metadata.get(key) == value for key, value in requested_filter.items()
                )
            metadata_rows.append(
                {
                    "question_id": trace["question_id"],
                    "strategy": trace["strategy"],
                    "rank": hit["rank"],
                    "chunk_id": hit["chunk_id"],
                    "unit_id": unit_id,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    "metadata_fields_match_source": all(field_results.values()),
                    "metadata_field_checks_json": json.dumps(
                        field_results, ensure_ascii=False, sort_keys=True
                    ),
                    "requested_filter_json": json.dumps(
                        requested_filter, ensure_ascii=False, sort_keys=True
                    )
                    if requested_filter
                    else "",
                    "filter_compliant": filter_compliant,
                    "manual_metadata_correct": "",
                    "manual_metadata_useful": "",
                    "manual_notes": "",
                }
            )

        grounding_rows.append(
            {
                "question_id": trace["question_id"],
                "strategy": trace["strategy"],
                "question": trace["question"],
                "top3_evidence_hit_auto": trace["evidence_hit_at_3"],
                "top3_chunks_json": json.dumps(
                    [
                        {
                            "rank": hit["rank"],
                            "chunk_id": hit["chunk_id"],
                            "text": hit["text"],
                            "score": hit["score"],
                            "metadata": hit["metadata"],
                        }
                        for hit in trace["hits"][:TOP_K]
                    ],
                    ensure_ascii=False,
                ),
                "manual_grounding_score_0_to_2": "",
                "manual_reason": "",
            }
        )

    metadata_audit_path = results_directory / "metadata_audit.csv"
    _preserve_existing_manual_fields(
        metadata_audit_path,
        metadata_rows,
        ("question_id", "strategy", "rank", "chunk_id"),
        ("manual_metadata_correct", "manual_metadata_useful", "manual_notes"),
    )
    _write_csv(
        metadata_audit_path,
        list(metadata_rows[0])
        if metadata_rows
        else [
            "question_id",
            "strategy",
            "rank",
            "chunk_id",
            "unit_id",
            "metadata_json",
            "metadata_fields_match_source",
            "metadata_field_checks_json",
            "requested_filter_json",
            "filter_compliant",
            "manual_metadata_correct",
            "manual_metadata_useful",
            "manual_notes",
        ],
        metadata_rows,
    )
    grounding_audit_path = results_directory / "grounding_audit.csv"
    _preserve_existing_manual_fields(
        grounding_audit_path,
        grounding_rows,
        ("question_id", "strategy"),
        ("manual_grounding_score_0_to_2", "manual_reason"),
    )
    _write_csv(
        grounding_audit_path,
        list(grounding_rows[0])
        if grounding_rows
        else [
            "question_id",
            "strategy",
            "question",
            "top3_evidence_hit_auto",
            "top3_chunks_json",
            "manual_grounding_score_0_to_2",
            "manual_reason",
        ],
        grounding_rows,
    )
    _create_coherence_audit(store, pages, results_directory / "coherence_audit.csv")

    valid_metadata_records = sum(
        bool(row["metadata_fields_match_source"]) for row in metadata_rows
    )
    total_metadata_records = len(metadata_rows)
    c_filter_records = [
        row
        for row in metadata_rows
        if row["strategy"] == "C" and row["filter_compliant"] is not None
    ]
    c_filter_questions = [
        trace
        for trace in test_traces
        if trace["strategy"] == "C" and trace.get("filter")
    ]
    c_filter_question_compliant = sum(
        bool(trace["hits"])
        and all(
            all(
                hit.get("metadata", {}).get(key) == value
                for key, value in trace["filter"].items()
            )
            for hit in trace["hits"][:TOP_K]
        )
        for trace in c_filter_questions
    )
    return {
        "test_top3_metadata": {
            "valid_records": valid_metadata_records,
            "total_records": total_metadata_records,
            "rate": (
                valid_metadata_records / total_metadata_records
                if total_metadata_records
                else None
            ),
            "method": "Automated comparison of returned metadata with the source corpus; manual audit labels are stored in separate columns.",
        },
        "strategy_C_test_filter_compliance": {
            "compliant_questions": c_filter_question_compliant,
            "questions_with_filter": len(c_filter_questions),
            "question_rate": (
                c_filter_question_compliant / len(c_filter_questions)
                if c_filter_questions
                else None
            ),
            "compliant_records": sum(bool(row["filter_compliant"]) for row in c_filter_records),
            "records_with_filter": len(c_filter_records),
            "record_rate": (
                sum(bool(row["filter_compliant"]) for row in c_filter_records)
                / len(c_filter_records)
                if c_filter_records
                else None
            ),
        },
        "manual_labeling": {
            "metadata_rows": len(metadata_rows),
            "grounding_rows": len(grounding_rows),
            "coherence_rows": 30,
            "labels_filled_by_pipeline": 0,
            "instruction": "The pipeline does not create subjective labels. Matching existing labels are preserved on rerun; new or changed audit rows remain blank. Do not treat automatic evidence matching as a human grounding score.",
        },
    }


def _update_experiment_metrics(summary: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "strategy",
        "chunking",
        "chunk_size",
        "overlap",
        "metadata",
        "filter",
        "vectorizer",
        "hypothesis",
        "dev_hit1",
        "dev_hit3",
        "test_hit1",
        "test_hit3",
        "notes",
    ]
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        existing = {
            row["strategy"]: row
            for row in reader
            if row.get("strategy") in STRATEGIES
        }
    for strategy in STRATEGIES:
        if strategy not in existing:
            raise ValueError(
                f"Missing Step 3 experiment row for strategy {strategy}; run src.run first"
            )
        for split, column in (("dev", "dev"), ("test", "test")):
            for metric, suffix in (("evidence_hit_at_1", "hit1"), ("evidence_hit_at_3", "hit3")):
                result = summary[strategy][split][metric]
                existing[strategy][f"{column}_{suffix}"] = (
                    f"{result['hits']}/{result['total']} "
                    f"({result['rate']:.1%})"
                    if result["total"]
                    else "N/A"
                )

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in STRATEGIES:
            writer.writerow(existing[strategy])
    temporary_path.replace(path)


def main() -> None:
    load_dotenv(ROOT / ".env")
    provider = os.getenv("EMBEDDING_PROVIDER", "chroma-local")
    if provider != "chroma-local":
        raise ValueError("This benchmark runner supports EMBEDDING_PROVIDER=chroma-local only")
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = ChromaLocalEmbedder(model_name)
    persist_directory = _configured_path(
        os.getenv("CHROMA_PERSIST_DIRECTORY"),
        ROOT / "data" / "chroma",
    )

    pages = load_corpus_units(ROOT / "data" / "corpus_units.jsonl")
    units = {str(page.metadata["unit_id"]): page for page in pages}
    questions = _load_questions(ROOT / "data" / "benchmark.jsonl")
    _validate_questions(questions, units)
    store = ChromaVectorStore(persist_directory, embedder.dimensions)

    for strategy in STRATEGIES:
        expected_count = store.count(strategy)
        if expected_count == 0:
            raise RuntimeError(f"Strategy {strategy} has no indexed chunks; run `python -m src.run`")

    run_id = datetime.now().astimezone().isoformat(timespec="seconds")
    traces: list[dict[str, Any]] = []
    query_embedding_seconds = 0.0
    search_seconds: dict[str, float] = {strategy: 0.0 for strategy in STRATEGIES}
    for question in questions:
        embedding_start = perf_counter()
        query_vector = embedder.embed_query(question["question"])
        query_embedding_elapsed = perf_counter() - embedding_start
        query_embedding_seconds += query_embedding_elapsed
        for strategy in STRATEGIES:
            requested_filter = question.get("filter") or None
            applied_filter = requested_filter if strategy == "C" else None
            search_start = perf_counter()
            hits = store.search(strategy, query_vector, top_k=TOP_K, filters=applied_filter)
            elapsed = perf_counter() - search_start
            search_seconds[strategy] += elapsed
            result = evaluate_query(question, hits)
            traces.append(
                {
                    "run_id": run_id,
                    "strategy": strategy,
                    "question_id": question["id"],
                    "split": question["split"],
                    "type": question.get("type"),
                    "question": question["question"],
                    "expected_answer": question["expected_answer"],
                    "gold_source_id": question["gold_source_id"],
                    "gold_page": question["gold_page"],
                    "gold_unit_id": question.get("gold_unit_id", question["gold_page"]),
                    "gold_logical_page": question.get("gold_logical_page"),
                    "evidence_quote": question["evidence_quote"],
                    "filter_requested": requested_filter,
                    "filter": applied_filter,
                    "filter_context": question.get("filter_context"),
                    "query_embedding_seconds": query_embedding_elapsed,
                    "search_seconds": elapsed,
                    **result,
                }
            )

    metric_summary = summarize(traces)
    results_directory = ROOT / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    trace_path = results_directory / "retrieval_traces.jsonl"
    _write_jsonl(trace_path, traces)
    human_audits = _create_human_audits(traces, pages, store, results_directory)

    summary = {
        "run_id": run_id,
        "python_version": platform.python_version(),
        "chromadb_version": chromadb.__version__,
        "embedding_provider": provider,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "Chroma PersistentClient",
        "vector_store_path": str(persist_directory),
        "distance_metric": "cosine",
        "search_type": "Chroma HNSW approximate nearest-neighbor",
        "corpus_units": len(pages),
        "benchmark_questions": len(questions),
        "split_counts": {
            split: sum(question["split"] == split for question in questions)
            for split in ("dev", "test")
        },
        "top_k": TOP_K,
        "selection_rule": "Select by dev Evidence Hit@3, then dev Evidence Hit@1, then manual grounding/metadata and resource use. Test was not used for tuning.",
        "evidence_metric_definition": "Strict automatic proxy: the retrieved chunk must match a gold document/unit and contain the normalized evidence quote verbatim. It may undercount equivalent evidence requiring human confirmation; manual grounding is recorded separately.",
        "timing": {
            "query_embeddings_total_seconds": query_embedding_seconds,
            "query_embeddings_reused_across_strategies": True,
            "vector_search_total_seconds_by_strategy": search_seconds,
        },
        "retrieval_metrics": metric_summary,
        "metadata_and_filter_audit": human_audits,
        "manual_review_status": "Review any blank subjective audit fields; matching existing labels are preserved when rerunning evaluation. Disclose AI assistance where applicable.",
        "trace_file": str(trace_path.relative_to(ROOT)),
        "audit_files": [
            "metadata_audit.csv",
            "grounding_audit.csv",
            "coherence_audit.csv",
        ],
    }
    summary_path = results_directory / "retrieval_metrics.json"
    temporary_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_summary_path.replace(summary_path)
    _update_experiment_metrics(metric_summary, REPO_ROOT / "artifacts" / "experiment_log.csv")

    for strategy in STRATEGIES:
        for split in ("dev", "test"):
            metrics = metric_summary[strategy][split]
            hit1 = metrics["evidence_hit_at_1"]
            hit3 = metrics["evidence_hit_at_3"]
            print(
                f"{strategy} {split}: Evidence Hit@1={hit1['hits']}/{hit1['total']} "
                f"({hit1['rate']:.1%}); Evidence Hit@3={hit3['hits']}/{hit3['total']} "
                f"({hit3['rate']:.1%})"
            )
    filter_result = human_audits["strategy_C_test_filter_compliance"]
    print(
        "C test filter compliance: "
        f"{filter_result['compliant_questions']}/{filter_result['questions_with_filter']} "
        "filtered questions"
    )
    print(f"Trace: {trace_path}")
    print(f"Metrics: {summary_path}")
    print(f"Manual audit sheets: {results_directory}")
    print("Human audit scores remain blank; review them before claiming manual evaluation complete.")


if __name__ == "__main__":
    main()
