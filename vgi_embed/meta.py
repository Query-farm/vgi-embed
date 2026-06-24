"""Per-object discovery/metadata tags shared by the embed scalars and tables.

VGI 0.23.0 (strict profile) gates a handful of per-object tags on *every*
function and table (and on the catalog/schema). This module builds them in one
place so the surface stays consistent:

- ``vgi.title`` (VGI124)        -- human-friendly display name (must NOT
  normalize-equal the machine name, or VGI125 fires).
- ``vgi.description_llm`` (VGI112) -- Markdown narrative aimed at an LLM/agent.
- ``vgi.description_md`` (VGI113)  -- Markdown narrative aimed at human docs.
- ``vgi.keywords`` (VGI126)        -- comma-separated search terms/synonyms.
- ``vgi.source_url`` (VGI128)      -- link to the implementing source file.
"""

from __future__ import annotations

_REPO = "https://github.com/Query-farm/vgi-embed"


def source_url(relative_path: str) -> str:
    """Build a ``vgi.source_url`` for a file under the repo root on ``main``."""
    return f"{_REPO}/blob/main/{relative_path}"


def object_tags(
    *,
    title: str,
    description_llm: str,
    description_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Assemble the per-object VGI124/112/113/126/128 tag set."""
    return {
        "vgi.title": title,
        "vgi.description_llm": description_llm,
        "vgi.description_md": description_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
