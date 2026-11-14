"""Parameter validation and presentation; the worker and SQL own the analysis."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display


def configure(top_k: int):
    if type(top_k) is not int or not 1 <= top_k <= 8:
        raise ValueError("TOP_K must be an integer from 1 through 8 substantive passages")
    previous = os.environ.get("VGI_EMBED_DEFAULT_MODEL")
    os.environ["VGI_EMBED_DEFAULT_MODEL"] = "BAAI/bge-small-en-v1.5"
    return previous, datetime.now(UTC).isoformat()


def cleanup(previous):
    if previous is None:
        os.environ.pop("VGI_EMBED_DEFAULT_MODEL", None)
    else:
        os.environ["VGI_EMBED_DEFAULT_MODEL"] = previous


def prose(text: str) -> None:
    display(Markdown(text))


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.08, 0.90, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.82))
    fig.text(0.08, 0.03, source, fontsize=8, color="#6C7D73")
    plt.show()
