# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval helpers: per-source chunk diversity and collection catalog injection."""

from __future__ import annotations

import logging
import re
import os
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from nvidia_rag.utils.vdb.vdb_base import VDBBase

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp"})
SMALL_FILE_EXTENSIONS = frozenset(
    {".sh", ".py", ".log", ".out", ".bat", ".java", ".f", ".in", ".dat", ".nc", ".radar"}
)

OVERVIEW_QUERY_RE = re.compile(
    r"\b("
    r"summarize\s+(the\s+)?(data\s*set|dataset|collection|corpus|files?|documents?)"
    r"|list\s+(all\s+)?(files?|documents?)"
    r"|how\s+many\s+(files?|documents?)"
    r"|file\s+count"
    r"|what\s+files?"
    r"|overview\s+of\s+(the\s+)?(collection|dataset|data)"
    r"|describe\s+(the\s+)?(collection|dataset|corpus)"
    r")\b",
    re.IGNORECASE,
)


def _env_bool(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def document_basename(doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source", {})
    source_path = source.get("source_name", "") if isinstance(source, dict) else source
    if not source_path:
        return "unknown"
    return os.path.basename(str(source_path))


def is_collection_overview_query(query: str) -> bool:
    return bool(OVERVIEW_QUERY_RE.search(query or ""))


def diversify_documents_by_source(
    documents: list[Document],
    *,
    max_per_source: int | None = None,
    target_total: int | None = None,
) -> list[Document]:
    """Round-robin cap chunks per file so large logs do not dominate context."""
    if not documents:
        return documents
    if not _env_bool("APP_RETRIEVER_DIVERSIFY", True):
        return documents

    max_per_source = max_per_source or _env_int("APP_RETRIEVER_MAX_PER_SOURCE", 2)
    target_total = target_total or len(documents)

    by_source: dict[str, deque[Document]] = defaultdict(deque)
    source_order: list[str] = []
    for doc in documents:
        name = document_basename(doc)
        if name not in by_source:
            source_order.append(name)
        by_source[name].append(doc)

    selected: list[Document] = []
    counts: dict[str, int] = defaultdict(int)
    while len(selected) < target_total:
        progressed = False
        for name in source_order:
            if len(selected) >= target_total:
                break
            if counts[name] >= max_per_source:
                continue
            if by_source[name]:
                selected.append(by_source[name].popleft())
                counts[name] += 1
                progressed = True
        if not progressed:
            break
    logger.info(
        "Diversified retrieval: %d chunks from %d files (max %d/source, target %d)",
        len(selected),
        len(counts),
        max_per_source,
        target_total,
    )
    return selected


def supplement_underrepresented_files(
    documents: list[Document],
    vdb_op: "VDBBase",
    collection_names: list[str],
    *,
    target_total: int | None = None,
) -> list[Document]:
    """Add at least one chunk from images and small files missing after diversity."""
    if not documents or not collection_names:
        return documents
    if not _env_bool("APP_RETRIEVER_SUPPLEMENT_MISSING", True):
        return documents

    target_total = target_total or len(documents)
    represented = {document_basename(d) for d in documents}
    supplemented = list(documents)

    for collection_name in collection_names:
        try:
            catalog = vdb_op.get_documents(collection_name)
        except Exception as exc:
            logger.warning("Could not load catalog for %s: %s", collection_name, exc)
            continue

        priority_names: list[str] = []
        for entry in catalog:
            name = entry.get("document_name", "")
            if not name or name in represented:
                continue
            ext = os.path.splitext(name)[1].lower()
            doc_type = (entry.get("document_info") or {}).get("document_type", "")
            if ext in IMAGE_EXTENSIONS or doc_type in {"pdf", "image"}:
                priority_names.append(name)
            elif ext in SMALL_FILE_EXTENSIONS:
                priority_names.append(name)

        for name in priority_names:
            if len(supplemented) >= target_total:
                break
            try:
                chunks = vdb_op.retrieve_chunks_by_source_basename(collection_name, name, limit=1)
            except Exception:
                chunks = []
            if chunks:
                supplemented.append(chunks[0])
                represented.add(name)
                logger.info("Supplemented missing file into context: %s", name)

    return supplemented[:target_total]


def build_collection_catalog_text(
    vdb_op: "VDBBase",
    collection_names: list[str],
) -> str:
    """Authoritative file inventory for correct counts and file-type awareness."""
    if not _env_bool("APP_INJECT_COLLECTION_CATALOG", True):
        return ""

    sections: list[str] = []
    for collection_name in collection_names:
        try:
            docs = vdb_op.get_documents(collection_name)
        except Exception as exc:
            logger.warning("Catalog build failed for %s: %s", collection_name, exc)
            continue
        if not docs:
            continue

        lines = [
            f"COLLECTION INVENTORY ({collection_name}): {len(docs)} indexed files total.",
            "Use this inventory for file counts and filenames. Each file type below is equally relevant.",
        ]
        for idx, entry in enumerate(sorted(docs, key=lambda d: d.get("document_name", "")), 1):
            name = entry.get("document_name", "unknown")
            info = entry.get("document_info") or {}
            doc_type = info.get("document_type", "unknown")
            elements = info.get("total_elements", "?")
            desc = (info.get("description") or "").strip()
            if len(desc) > 140:
                desc = desc[:137] + "..."
            line = f"{idx}. {name} (type={doc_type}, elements={elements})"
            if desc:
                line += f" — {desc}"
            lines.append(line)
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return "\n\n".join(sections)


def should_inject_catalog(query: str, collection_names: list[str], vdb_op: "VDBBase") -> bool:
    if is_collection_overview_query(query):
        return True
    if not _env_bool("APP_INJECT_COLLECTION_CATALOG_ON_MULTI_FILE", True):
        return False
    for collection_name in collection_names:
        try:
            if len(vdb_op.get_documents(collection_name)) > 5:
                return True
        except Exception:
            continue
    return False
