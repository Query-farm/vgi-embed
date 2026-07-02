"""Per-object discovery/metadata tags shared by the embed scalars and tables.

VGI 0.23.0 (strict profile) gates a handful of per-object tags on *every*
function and table (and on the catalog/schema). This module builds them in one
place so the surface stays consistent:

- ``vgi.title`` (VGI124)    -- human-friendly display name (must NOT
  normalize-equal the machine name, or VGI125 fires).
- ``vgi.doc_llm`` (VGI112)  -- Markdown narrative aimed at an LLM/agent.
- ``vgi.doc_md`` (VGI113)   -- Markdown narrative aimed at human docs.
- ``vgi.keywords`` (VGI126/VGI138) -- a JSON array of search terms/synonyms
  (``["a", "b"]``); the comma-separated form is no longer accepted.

``vgi.source_url`` is *not* emitted per object: provenance lives on the catalog
(``Catalog(source_url=...)``); repeating it on every object trips VGI139.
"""

from __future__ import annotations

import json


def keywords_json(keywords: list[str]) -> str:
    """Serialize keywords as a ``vgi.keywords`` JSON array string."""
    return json.dumps(keywords)


def object_tags(
    *,
    title: str,
    description_llm: str,
    description_md: str,
    keywords: list[str],
    category: str | None = None,
) -> dict[str, str]:
    """Assemble the per-object VGI124/112/113/126/138 tag set.

    ``category`` (VGI411) names one of the schema's ``vgi.categories`` registry
    entries; every function/view carries exactly one so the worker's navigation
    and SEO listing sections stay populated.
    """
    tags = {
        "vgi.title": title,
        "vgi.doc_llm": description_llm,
        "vgi.doc_md": description_md,
        "vgi.keywords": keywords_json(keywords),
    }
    if category is not None:
        tags["vgi.category"] = category
    return tags
