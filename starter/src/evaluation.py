"""Retrieval scoring and metric aggregation for the Day 06 benchmark."""
import unicodedata
from typing import Any


def _normalize_evidence(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split()).casefold()


def _gold_locations(question: dict[str, Any]) -> set[tuple[str, str]]:
    locations = {
        (
            str(question["gold_source_id"]),
            str(question.get("gold_unit_id") or question["gold_page"]),
        )
    }
    for alternative in question.get("gold_alternatives", []):
        locations.add(
            (
                str(alternative["gold_source_id"]),
                str(alternative.get("gold_unit_id") or alternative["gold_page"]),
            )
        )
    return locations


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = _normalize_evidence(str(question["evidence_quote"]))
    gold_locations = _gold_locations(question)
    evaluated_hits: list[dict[str, Any]] = []

    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata") or {}
        location = (
            str(metadata.get("document_id", "")),
            str(metadata.get("unit_id", metadata.get("page", ""))),
        )
        location_match = location in gold_locations
        evidence_match = bool(evidence) and evidence in _normalize_evidence(
            str(hit.get("text") or "")
        )
        evaluated_hits.append(
            {
                **hit,
                "rank": rank,
                "gold_location_match": location_match,
                "evidence_exact_match": evidence_match,
                "evidence_hit": location_match and evidence_match,
            }
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "gold_location": [
            {"document_id": source_id, "unit_id": unit_id}
            for source_id, unit_id in sorted(gold_locations)
        ],
        "evidence_hit_at_1": bool(evaluated_hits and evaluated_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in evaluated_hits[:3]),
        "hits": evaluated_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = sorted({str(trace["strategy"]) for trace in traces})
    summary: dict[str, Any] = {}
    for strategy in strategies:
        strategy_traces = [trace for trace in traces if trace["strategy"] == strategy]
        summary[strategy] = {}
        for split in ("dev", "test", "all"):
            split_traces = [
                trace
                for trace in strategy_traces
                if split == "all" or trace["split"] == split
            ]
            denominator = len(split_traces)
            hit_at_1 = sum(bool(trace["evidence_hit_at_1"]) for trace in split_traces)
            hit_at_3 = sum(bool(trace["evidence_hit_at_3"]) for trace in split_traces)
            summary[strategy][split] = {
                "questions": denominator,
                "evidence_hit_at_1": {
                    "hits": hit_at_1,
                    "total": denominator,
                    "rate": hit_at_1 / denominator if denominator else None,
                },
                "evidence_hit_at_3": {
                    "hits": hit_at_3,
                    "total": denominator,
                    "rate": hit_at_3 / denominator if denominator else None,
                },
            }

    filter_questions = [
        trace for trace in traces if trace["strategy"] == "C" and trace.get("filter")
    ]
    filter_compliant = sum(
        bool(trace["hits"])
        and all(
            (hit.get("metadata") or {}).get(key) == value
            for hit in trace["hits"]
            for key, value in trace["filter"].items()
        )
        for trace in filter_questions
    )
    summary.setdefault("C", {})["filter_compliance"] = {
        "questions_compliant": filter_compliant,
        "questions_with_filter": len(filter_questions),
        "rate": filter_compliant / len(filter_questions) if filter_questions else None,
    }
    return summary
