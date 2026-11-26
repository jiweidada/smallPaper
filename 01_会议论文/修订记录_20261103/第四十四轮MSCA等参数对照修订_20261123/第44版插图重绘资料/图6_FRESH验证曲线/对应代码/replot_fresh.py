"""Reproduce Figure 6 from the two archived FRESH validation series (percent)."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGURE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = FIGURE_DIR / "图6_FRESH逐轮绘图数据_百分数.csv"
ARMS = ("BASE_FRESH", "FULL_FRESH")
MEASURES = ("AP", "APs")
COLORS = {"BASE_FRESH": "#0072B2", "FULL_FRESH": "#D55E00"}
LINESTYLES = {"BASE_FRESH": "--", "FULL_FRESH": "-"}


def read_percentages(path):
    """Read both arms in round order. The four metric columns are already percentages."""
    columns = [f"{arm}_{metric}_pct" for metric in MEASURES for arm in ARMS]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(["round"] + columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        by_round = {}
        for row in reader:
            number = int(row["round"])
            if number in by_round:
                raise ValueError(f"Duplicate training round {number}")
            values = {name: float(row[name]) for name in columns}
            if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values.values()):
                raise ValueError(f"Invalid percentage at training round {number}")
            by_round[number] = values
    if sorted(by_round) != list(range(1, 321)):
        raise ValueError("Expected a complete series from round 1 to round 320")
    return {number: by_round[number] for number in sorted(by_round)}


def make_figure(rows):
    rounds = sorted(rows)
    # Both panels use each arm's highest overall AP round. Sorted order breaks ties at the earliest round.
    best_rounds = {
        arm: max(rounds, key=lambda number: rows[number][f"{arm}_AP_pct"])
        for arm in ARMS
    }
    settings = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}
    with plt.rc_context(settings):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 3))
        for axis, metric, title, ylabel, top in zip(
            axes, MEASURES, ("Validation AP", "Small-object validation AP"),
            ("AP (%)", r"AP$_{s}$ (%)"), (25, 18)
        ):
            axis.axvspan(296.5, 320.5, color="#E9E9E9", zorder=0)
            for arm in ARMS:
                name = f"{arm}_{metric}_pct"
                axis.plot(rounds, [rows[number][name] for number in rounds],
                          color=COLORS[arm], linestyle=LINESTYLES[arm], linewidth=1.0,
                          label=arm, zorder=2)
                for number, symbol, area in [(320, "o", 22), (best_rounds[arm], "*", 65)]:
                    axis.scatter(number, rows[number][name], color=COLORS[arm],
                                 marker=symbol, s=area, edgecolors="black", linewidths=0.45, zorder=4)
            axis.set(title=title, xlabel="Completed training round", ylabel=ylabel,
                     xlim=(0, 326), ylim=(0, top), xticks=[0, 80, 160, 240, 320])
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        handles = [Line2D([], [], color=COLORS[arm], linestyle=LINESTYLES[arm],
                          linewidth=1.2, label=arm) for arm in ARMS]
        for label, symbol, size in [("Round 320", "o", 4), ("Best-AP round", "*", 6)]:
            handles.append(Line2D([], [], linestyle="none", color="#707070",
                                  marker=symbol, markersize=size, label=label))
        figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                      fontsize=8, bbox_to_anchor=(0.5, 0.005))
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.88, wspace=0.30)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=FIGURE_DIR / "图6_FRESH曲线_复现.png")
    args = parser.parse_args()
    figure = make_figure(read_percentages(args.csv))
    try:
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            figure.savefig(args.output, dpi=320, facecolor="white")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()

