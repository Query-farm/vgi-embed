"""Plot SQL-returned embedding scores and bounded review populations."""

import matplotlib.pyplot as plt
import numpy as np
from vgi_notebooks import finish

SOURCE = (
    "Authored text and labels · VGI Embed v0.5.0 · pinned Qdrant BGE revision aa8f8b0\n"
    "Local model inference · no measured retrieval or duplicate-detection benchmark"
)


def ranking(ranking, TOP_K):
    """Display returned scores with their explicit selection denominator."""
    fig = plt.figure(figsize=(14, 6.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.12, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    for ax, (_, part) in zip(axes, ranking.groupby("query_id", sort=True), strict=True):
        y = np.arange(len(part))
        colors = ["#2A7857" if v else "#9EA9A3" for v in part.relevant]
        ax.hlines(y, 0, part.score, color=colors, linewidth=3, alpha=0.5)
        for relevant, marker in [(True, "o"), (False, "x")]:
            mask = part.relevant.to_numpy() == relevant
            ax.scatter(
                part.score.to_numpy()[mask],
                y[mask],
                color="#2A7857" if relevant else "#7D8982",
                marker=marker,
                s=55,
                label="Judged relevant" if relevant else "Other passage",
                zorder=3,
            )
        for pos, row in zip(y, part.itertuples(), strict=True):
            ax.annotate(
                f"{row.score:.3f}",
                (row.score, pos),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, [f"{r.passage_id}  {r.short_name}" for r in part.itertuples()], fontsize=9)
        ax.invert_yaxis()
        if len(part) > TOP_K:
            ax.axhline(TOP_K - 0.5, color="#C08A32", linestyle="--", linewidth=1.5)
        ax.set_title(part.topic.iloc[0], loc="left", fontsize=12, pad=10)
        ax.set_xlabel("Cosine similarity (dimensionless)")
        ax.set_xlim(min(0, ranking.score.min() - 0.05), max(0.9, ranking.score.max() + 0.12))
    axes[0].legend(loc="lower right", fontsize=8)
    finish(
        fig,
        f"Return the first {TOP_K} passages for each question",
        "Exhaustive ranking · authored relevance judgments · dashed line marks the cutoff",
        SOURCE,
        name="ranking",
    )


def tradeoff(curve_data, TOP_K):
    """Display returned scores with their explicit selection denominator."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7), sharex=True, sharey=True)
    for ax, (_, part) in zip(axes, curve_data.groupby("query_id", sort=True), strict=True):
        ax.plot(part.k, part.recall_at_k * 100, "o-", color="#2A7857", label="Recall", linewidth=2)
        ax.plot(part.k, part.precision_at_k * 100, "s--", color="#B47722", label="Precision", linewidth=2)
        chosen = part[part.k == TOP_K].iloc[0]
        ax.axvline(TOP_K, color="#65776C", linestyle=":", linewidth=1.5)
        ax.scatter(
            [TOP_K, TOP_K],
            [chosen.recall_at_k * 100, chosen.precision_at_k * 100],
            s=140,
            facecolors="none",
            edgecolors="#243B2F",
            linewidths=1.4,
            zorder=4,
        )
        ax.set_title(part.topic.iloc[0], loc="left", fontsize=12, pad=12)
        ax.set_xticks(range(1, 9))
        ax.set_ylim(-4, 108)
        ax.set_xlabel("Passages returned per question (k)")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Share of the applicable denominator (%)")
    finish(
        fig,
        "More results change coverage and reading effort",
        f"Selected k = {TOP_K} · recall denominator: all judged relevant passages · "
        "precision denominator: k returned passages",
        SOURCE,
        name="tradeoff",
    )


def matrix(data):
    """Show mirrored stored pair scores, with the diagonal explicitly contextualized."""
    frame = data.pivot(index="left_id", columns="right_id", values="score")
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    img = ax.imshow(frame.to_numpy(), vmin=-1, vmax=1, cmap="BrBG")
    ax.set_xticks(range(len(frame)), frame.columns)
    ax.set_yticks(range(len(frame)), frame.index)
    ax.set_xlabel("Title ID")
    ax.set_ylabel("Title ID")
    ax.set_xticks(np.arange(-0.5, len(frame), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(frame), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for i in range(len(frame)):
        for j in range(len(frame)):
            v = frame.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(v) > 0.65 else "#243b2f",
            )
    fig.colorbar(img, ax=ax, label="Cosine similarity (dimensionless)", shrink=0.85)
    finish(
        fig,
        "Similar wording is a review signal",
        "Seven usable titles · 21 unordered scores mirrored · diagonal: self-reference 1",
        SOURCE,
        name="title-pair-map",
    )


def counts(data, threshold):
    """Reconcile flagged and below-threshold pairs without dropping zero categories."""
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = np.arange(len(data))
    ax.barh(x, data.flagged, color="#397d62", label="Flagged for review")
    ax.barh(x, data.below_threshold, left=data.flagged, color="#d8e1dc", label="Below threshold")
    for i, r in enumerate(data.itertuples()):
        ax.text(
            r.all_pairs + 0.2, i, f"{r.flagged} / {r.all_pairs} flagged", ha="left", va="center", fontsize=10
        )
    ax.set_yticks(x, data.role)
    ax.invert_yaxis()
    ax.set_xlim(0, max(data.all_pairs) + 4)
    ax.set_xlabel("Unordered title pairs (count)")
    ax.set_xticks(range(0, int(max(data.all_pairs)) + 1, 2))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    finish(
        fig,
        f"Threshold {threshold:.2f}: keep the full pair population",
        "Author-defined roles · zero flagged categories retained · no automatic merging",
        SOURCE,
        name="pair-review-counts",
    )
