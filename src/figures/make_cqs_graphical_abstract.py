from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path("article/submission_package/cqs_rank_latex_source_20260612/figures")


def add_box(ax, xy, width, height, title, lines, face, edge):
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.8,
            edgecolor=edge,
            facecolor=face,
            joinstyle="round",
        )
    )
    ax.text(
        x + 0.22,
        y + height - 0.42,
        title,
        ha="left",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        color="#17202a",
    )
    ax.text(
        x + 0.22,
        y + height - 1.05,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10.4,
        color="#243447",
        linespacing=1.15,
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.2,
            color="#50616b",
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(
        0.45,
        6.52,
        "CQS-Rank calibrated queue-set evidential graph ranking",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
        color="#17202a",
    )
    ax.text(
        0.45,
        6.02,
        "Calibrated scheduled-arrival risk remains auditable; event mass selects a fixed-capacity airport-hour review queue.",
        fontsize=14,
        ha="left",
        va="top",
        color="#394b59",
    )

    boxes = [
        (
            (0.55, 3.65),
            3.0,
            1.75,
            "Public records",
            ["weather, demand", "FAA advisories", "delayed closure"],
            "#e8f1f5",
            "#74858c",
        ),
        (
            (4.25, 3.65),
            3.0,
            1.75,
            "GT-AFRE anchor",
            ["fuzzy residuals", "graph evidence", "temporal memory"],
            "#e6f3ed",
            "#6d887c",
        ),
        (
            (7.95, 3.65),
            3.0,
            1.75,
            "CQS layer",
            ["closed-form score", "lower-bound score", "fixed queue"],
            "#efeaf5",
            "#82718e",
        ),
        (
            (11.65, 3.65),
            3.25,
            1.75,
            "Audit record",
            ["calibrated risk", "queue membership", "closed outcomes"],
            "#f1f4e6",
            "#7e8a68",
        ),
    ]
    for xy, width, height, title, lines, face, edge in boxes:
        add_box(ax, xy, width, height, title, lines, face, edge)

    for start_x in [3.58, 7.28, 10.98]:
        add_arrow(ax, (start_x, 4.48), (start_x + 0.44, 4.48))

    evidence = [
        (
            1.05,
            "Fixed-budget gain",
            ["Top-10 capture gains", "+0.0603 / +0.0474 / +0.0406"],
            "#ffffff",
        ),
        (
            5.0,
            "Carrier audit",
            ["positive vs GT/RAEG", "+0.0612 / +0.0486 / +0.0397"],
            "#ffffff",
        ),
        (
            9.35,
            "Robust certificate",
            ["calibration-gap penalty", "+9125 / +3136 / +1002"],
            "#ffffff",
        ),
    ]
    for x, title, lines, face in evidence:
        add_box(ax, (x, 1.08), 3.3, 1.75, title, lines, face, "#9aa7ad")

    ax.text(
        8.0,
        0.34,
        "The method turns heterogeneous public records into calibrated risk and a calibration-robust fixed-capacity review set.",
        ha="center",
        va="center",
        fontsize=13,
        color="#17202a",
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"graphical_abstract_cqs_rank.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
