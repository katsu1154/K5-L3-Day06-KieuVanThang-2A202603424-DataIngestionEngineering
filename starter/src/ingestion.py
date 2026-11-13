"""Load corpus units and normalize their text while retaining source locations."""
import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in normalized.split("\n"):
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        if line:
            current_lines.append(line)
        elif current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

    if current_lines:
        paragraphs.append(" ".join(current_lines))
    return "\n\n".join(paragraphs)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        status = "ok" if text else "empty"
        if not text:
            warnings.warn(
                f"No extractable text on page {page_number} of {path}",
                RuntimeWarning,
                stacklevel=2,
            )
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"extraction_status": status},
            )
        )
    return pages


def load_corpus_units(
    path: Path,
    minimum_units: int = 30,
    minimum_characters: int = 1500,
) -> list[DocumentPage]:
    if minimum_units < 1 or minimum_characters < 1:
        raise ValueError("minimum_units and minimum_characters must be positive")

    pages: list[DocumentPage] = []
    seen_units: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}")

            required = (
                "source_id",
                "unit_id",
                "source_path_or_url",
                "logical_page",
                "volume",
                "text",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(
                    f"Missing fields on line {line_number} of {path}: {', '.join(missing)}"
                )

            document_id = record["source_id"]
            unit_id = record["unit_id"]
            source = record["source_path_or_url"]
            volume = record["volume"]
            text = record["text"]
            if not all(isinstance(value, str) and value.strip() for value in (
                document_id,
                unit_id,
                source,
                volume,
                text,
            )):
                raise ValueError(f"Expected non-empty text fields on line {line_number} of {path}")

            unit_key = (document_id, unit_id)
            if unit_key in seen_units:
                raise ValueError(f"Duplicate corpus unit {unit_id!r} on line {line_number}")
            seen_units.add(unit_key)

            normalized = normalize_text(text)
            if len(normalized) < minimum_characters:
                raise ValueError(
                    f"Corpus unit {unit_id!r} has {len(normalized)} normalized characters; "
                    f"minimum is {minimum_characters}"
                )
            try:
                logical_page = int(record["logical_page"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid logical_page on line {line_number} of {path}"
                ) from error
            if logical_page < 1:
                raise ValueError(f"logical_page must be positive on line {line_number}")

            metadata: dict[str, Any] = {
                "unit_id": unit_id,
                "volume": volume,
            }
            for key in (
                "unit_order",
                "location_type",
                "epub_item_path",
                "paragraph_start",
                "paragraph_end",
                "normalized_start_char",
                "normalized_end_char",
                "offset_basis",
            ):
                value = record.get(key)
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=logical_page,
                    text=normalized,
                    metadata=metadata,
                )
            )

    if len(pages) < minimum_units:
        raise ValueError(
            f"Corpus contains {len(pages)} content units; at least {minimum_units} are required"
        )
    return pages
