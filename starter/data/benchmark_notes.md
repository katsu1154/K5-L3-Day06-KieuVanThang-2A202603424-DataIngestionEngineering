# Benchmark notes

- Questions are in English to match the English-language source.
- The split is fixed before any retrieval experiment: 15 dev questions and 5 test questions.
- To reduce direct source-section overlap, dev and test questions come from disjoint EPUB spine sections. The corpus is still a single book, so this is not a document-independent test and may have topic leakage.
- `gold_page` is the content-unit ID, not a physical page number. `gold_logical_page` is the corresponding local unit number; use `gold_unit_id` for source tracing.
- Filters use the source-grounded `volume` metadata. The `filter_context` field records that the volume was selected before retrieval; no gold answer or source ID is used as a filter.
- Evidence quotes were checked against `corpus_units.jsonl`. The editor's notes and the historical author's claims are treated as statements in this source, not independently verified facts.
- Selection criterion fixed before comparing results: prioritize dev Evidence Hit@3, then Hit@1, manual grounding/metadata quality, and resource use. Do not tune against the test split.
- Results are saved in `../results/retrieval_metrics.json` and `../results/retrieval_traces.jsonl`. A: dev 9/15 and test 4/5 Evidence Hit@3; B and C: dev 12/15 and test 5/5. B/C tie on retrieval; C applies the source-grounded `volume` filter and complies on all 6 filtered queries (2 test).
- The audit CSVs contain AI-assisted draft judgments that the student reports reviewing and confirming. The AI assistance remains disclosed; these are not independent/blind ratings. See `../results/HUMAN_AUDIT_GUIDE.md` and the individual report.
