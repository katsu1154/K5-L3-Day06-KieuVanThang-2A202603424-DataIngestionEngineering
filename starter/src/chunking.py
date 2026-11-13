"""Fixed-size and paragraph-aware chunking with source offsets."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _build_chunk(
    page: DocumentPage,
    start: int,
    end: int,
    index: int,
    strategy: str,
    paragraph_start: int,
    paragraph_end: int,
) -> Chunk:
    text = page.text[start:end]
    identity = "\0".join(
        (page.document_id, str(page.page), strategy, str(start), str(end), text)
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    metadata = {
        **page.metadata,
        "chunking": strategy,
        "chunk_index": index,
        "chunk_start_char": start,
        "chunk_end_char": end,
        "chunk_paragraph_start": paragraph_start,
        "chunk_paragraph_end": paragraph_end,
    }
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(
    page: DocumentPage,
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    starts = list(range(0, len(page.text), step))
    if len(starts) > 1 and len(page.text) - starts[-1] <= overlap:
        starts.pop()
    for index, start in enumerate(starts, start=1):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(
                _build_chunk(
                    page,
                    start,
                    end,
                    index,
                    "fixed_size",
                    0,
                    0,
                )
            )
    return chunks


def _split_long_paragraph(
    text: str,
    start: int,
    end: int,
    paragraph_number: int,
    max_size: int,
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    part_start = start
    while end - part_start > max_size:
        hard_end = part_start + max_size
        boundary = text.rfind(" ", part_start + max_size // 2, hard_end)
        if boundary <= part_start:
            boundary = hard_end
        else:
            boundary += 1
        spans.append((part_start, boundary, paragraph_number))
        part_start = boundary
    if part_start < end:
        spans.append((part_start, end, paragraph_number))
    return spans


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not page.text:
        return []

    paragraph_spans: list[tuple[int, int, int]] = []
    cursor = 0
    for paragraph_number, paragraph in enumerate(page.text.split("\n\n"), start=1):
        start = cursor
        end = start + len(paragraph)
        if paragraph.strip():
            paragraph_spans.extend(
                _split_long_paragraph(page.text, start, end, paragraph_number, max_size)
            )
        cursor = end + 2

    grouped: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None
    for start, end, paragraph_number in paragraph_spans:
        if current is not None and end - current[0] <= max_size:
            current = (current[0], end, current[2], paragraph_number)
        else:
            if current is not None:
                grouped.append(current)
            current = (start, end, paragraph_number, paragraph_number)
    if current is not None:
        grouped.append(current)

    chunks: list[Chunk] = []
    for index, (start, end, first_paragraph, last_paragraph) in enumerate(grouped, start=1):
        if page.text[start:end].strip():
            chunks.append(
                _build_chunk(
                    page,
                    start,
                    end,
                    index,
                    "paragraph",
                    first_paragraph,
                    last_paragraph,
                )
            )
    return chunks
