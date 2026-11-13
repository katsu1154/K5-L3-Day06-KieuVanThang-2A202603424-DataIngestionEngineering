# Manual audit instructions

The evaluation pipeline does not assign subjective labels. The saved CSVs currently contain draft judgments assisted by AI; the student reports reviewing and confirming them. The original AI assistance remains disclosed. Do not copy the automatic exact-match flag into a human score without checking context.

Identify AI-assisted labels in the reason/notes and in the individual report. Do not describe AI-generated draft labels as independent or blind human ratings; the student should review and own any submitted assessment.

## Chunk coherence: `coherence_audit.csv`

For each of the same 10 source locations, rate the displayed chunk for each strategy:

- `0`: cut mid-idea or missing context makes it hard to understand.
- `1`: partly understandable, but missing important context.
- `2`: self-contained and preserves the complete idea.

Add a short reason. The locations are fixed deterministically: the first paragraph of at least 80 characters from each of the 10 lowest logical unit IDs, with the paragraph midpoint used to select the covering chunk. The source paragraph and selected chunk are both included.

## Metadata usefulness: `metadata_audit.csv`

The file contains the actual top-3 hits for all five test questions and all three strategies. `metadata_fields_match_source` is an automated integrity check against the corpus; it is **not** a manual label. Review `metadata_json` and the referenced source, then fill:

- `manual_metadata_correct`: `yes` or `no` for whether source, unit/page and filter metadata are correct and sufficient to trace the chunk.
- `manual_metadata_useful`: `yes` or `no` for whether the metadata helps scope or trace retrieval.
- `manual_notes`: explain any error, usefulness, or limitation found.

Only strategy C applies the question's filter; the requested filter and automatic compliance result are in the same row. Discuss one useful field and one limitation in the report.

## Grounding: `grounding_audit.csv`

For each of the five test questions under each strategy, judge the complete top-3 evidence without treating the automatic metric as a human score:

- `0`: no evidence supports an answer.
- `1`: relevant evidence exists but is missing an important condition or context.
- `2`: evidence is sufficient to answer and can be traced to its source.

Fill `manual_grounding_score_0_to_2` and add a concise reason. These are evidence retrieval judgments, not LLM answer-accuracy scores.

After labeling, report the counts at each score and mean /2 for grounding, mean /2 for coherence, metadata correctness/usefulness, filter compliance, and notable failure examples. Record the human rater and possible bias in the individual report.
