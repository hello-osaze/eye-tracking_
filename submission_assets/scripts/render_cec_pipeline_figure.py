from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-eyebench")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


NAVY = "#173D7A"
BLUE = "#355C97"
EDGE = "#A8BEE1"
BG = "#FBFCFF"
TEXT_BOX = "#EEF4FF"
GAZE_BOX = "#EEF9F2"
LATENT_BOX = "#F6EEFF"
OUTPUT_BOX = "#FFF8E8"


def panel(ax, x, y, w, h, title=None):
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=2.2,
        edgecolor=EDGE,
        facecolor=BG,
        linestyle=(0, (7, 5)),
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + w / 2, y + h + 2.0, title, ha="center", va="bottom", fontsize=15, fontweight="bold", color=NAVY)


def box(ax, x, y, w, h, title, lines, facecolor, title_size=12, body_size=9):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=2.4,
        edgecolor=NAVY,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=NAVY)


def arrow(ax, start, end, color=NAVY, lw=2.8, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.4, 6.5), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 84)
    ax.axis("off")

    panel(ax, 5, 1, 63, 79)
    panel(ax, 72, 18, 24, 40)

    # Inputs
    box(ax, 10, 66, 22, 8.0, "Claim + context", [], TEXT_BOX, title_size=13)
    box(ax, 40, 66, 22, 8.0, "Eye movements", [], GAZE_BOX, title_size=13)

    # Two branches into fusion
    box(ax, 10, 52, 22, 9.0, "Text encoder", [], TEXT_BOX, title_size=13.0)
    box(ax, 40, 52, 22, 9.0, "Gaze features", [], GAZE_BOX, title_size=13.0)
    box(ax, 25, 39, 22, 9.0, "Fuse text and gaze", [], GAZE_BOX, title_size=12.4)

    # Latent evidence core
    box(ax, 22, 28, 28, 9.0, "Evidence scorer", [], LATENT_BOX, title_size=13.0)
    box(ax, 28, 19.5, 16, 7.0, "Evidence weights", [], TEXT_BOX, title_size=12.0)

    # Two outputs from the same latent weights
    box(ax, 8, 10.0, 22, 8.0, "Evidence summary", [], OUTPUT_BOX, title_size=12.0)
    box(ax, 42, 10.0, 22, 8.0, "Coverage vector", [], GAZE_BOX, title_size=12.0)
    box(ax, 27, 2.5, 18, 6.0, "CEC output", [], TEXT_BOX, title_size=12.0)

    # Benchmark path
    box(ax, 74.5, 45.5, 19, 7.8, "Text-only RoBERTa", [], TEXT_BOX, title_size=11.8)
    box(ax, 74.0, 33.5, 20, 8.0, "Late fusion", [], OUTPUT_BOX, title_size=12.4)
    box(ax, 78.0, 22.5, 12, 7.0, "Final score", [], GAZE_BOX, title_size=11.8)

    # Clean arrows
    arrow(ax, (21, 66), (21, 61))
    arrow(ax, (51, 66), (51, 61))
    arrow(ax, (21, 52), (34, 48))
    arrow(ax, (51, 52), (38, 48))
    arrow(ax, (36, 39), (36, 37))
    arrow(ax, (36, 28), (36, 26.5))
    arrow(ax, (32, 19.5), (19, 18.0))
    arrow(ax, (40, 19.5), (53, 18.0))
    arrow(ax, (19, 10.0), (32.5, 8.5))
    arrow(ax, (53, 10.0), (39.5, 8.5))

    # Only one cross-panel arrow, routed through whitespace.
    arrow(ax, (45, 5.5), (74.0, 37.5), connectionstyle="angle3,angleA=0,angleB=90")
    arrow(ax, (84, 45.5), (84, 41.5))
    arrow(ax, (84, 33.5), (84, 29.5))

    fig.savefig(FIGURE_DIR / "cec_pipeline.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "cec_pipeline.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
