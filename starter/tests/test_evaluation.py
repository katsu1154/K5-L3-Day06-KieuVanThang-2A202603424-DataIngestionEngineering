from pathlib import Path
import tempfile
import unittest

from src.evaluate import _preserve_existing_manual_fields
from src.evaluation import evaluate_query, summarize


class EvaluateQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q1",
            "split": "dev",
            "gold_source_id": "doc-1",
            "gold_page": "unit-1",
            "gold_unit_id": "unit-1",
            "evidence_quote": "The exact supporting evidence.",
        }

    def test_requires_gold_location_and_exact_evidence(self) -> None:
        hits = [
            {
                "chunk_id": "wrong-unit",
                "text": "The exact supporting evidence.",
                "score": 0.9,
                "metadata": {"document_id": "doc-1", "unit_id": "unit-2"},
            },
            {
                "chunk_id": "gold-unit",
                "text": "Context. The exact\nsupporting evidence. More text.",
                "score": 0.8,
                "metadata": {"document_id": "doc-1", "unit_id": "unit-1"},
            },
        ]

        result = evaluate_query(self.question, hits)

        self.assertFalse(result["evidence_hit_at_1"])
        self.assertTrue(result["evidence_hit_at_3"])
        self.assertFalse(result["hits"][0]["evidence_hit"])
        self.assertTrue(result["hits"][1]["evidence_hit"])

    def test_summary_separates_splits_and_checks_filters(self) -> None:
        traces = [
            {
                "strategy": "C",
                "split": "dev",
                "evidence_hit_at_1": True,
                "evidence_hit_at_3": True,
                "filter": {"volume": "Volume 1"},
                "hits": [{"metadata": {"volume": "Volume 1"}}],
            },
            {
                "strategy": "C",
                "split": "test",
                "evidence_hit_at_1": False,
                "evidence_hit_at_3": True,
                "filter": {"volume": "Volume 2"},
                "hits": [{"metadata": {"volume": "Volume 1"}}],
            },
        ]

        result = summarize(traces)

        self.assertEqual(result["C"]["dev"]["evidence_hit_at_1"]["hits"], 1)
        self.assertEqual(result["C"]["dev"]["evidence_hit_at_1"]["total"], 1)
        self.assertEqual(result["C"]["test"]["evidence_hit_at_3"]["hits"], 1)
        self.assertEqual(result["C"]["filter_compliance"]["questions_compliant"], 1)
        self.assertEqual(result["C"]["filter_compliance"]["questions_with_filter"], 2)

    def test_preserves_only_matching_existing_manual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "audit.csv"
            path.write_text(
                "question_id,chunk_id,score,reason\nq1,c1,2,Reviewed evidence\n",
                encoding="utf-8",
            )
            rows = [
                {"question_id": "q1", "chunk_id": "c1", "score": "", "reason": ""},
                {"question_id": "q1", "chunk_id": "c2", "score": "", "reason": ""},
            ]

            _preserve_existing_manual_fields(
                path,
                rows,
                ("question_id", "chunk_id"),
                ("score", "reason"),
            )

        self.assertEqual(rows[0]["score"], "2")
        self.assertEqual(rows[0]["reason"], "Reviewed evidence")
        self.assertEqual(rows[1]["score"], "")
        self.assertEqual(rows[1]["reason"], "")


if __name__ == "__main__":
    unittest.main()
