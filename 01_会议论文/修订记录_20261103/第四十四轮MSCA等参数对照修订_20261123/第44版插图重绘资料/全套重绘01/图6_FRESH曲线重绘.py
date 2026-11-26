"""Redraw both FRESH panels from the revision 44 CSV and its archived plotting functions."""
import argparse
import hashlib
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

INK = "#223D4B"
MUTED = "#405968"
GRID = "#D5E0E1"
SHADE = "#E2EBEC"
HERE = Path(__file__).resolve().parent
DEFAULT_RESOURCES = HERE.parent / "图6_FRESH验证曲线"

def draw(resources, destination, stem):
    archive = runpy.run_path(str(resources / "对应代码" / "replot_fresh.py"))
    csv_path = resources / "图6_FRESH逐轮绘图数据_百分数.csv"
    rows = archive["read_percentages"](csv_path)
    arms = archive["ARMS"]
    colors = archive["COLORS"]
    styles = archive["LINESTYLES"]
    peaks = {arm: max(sorted(rows), key=lambda r: rows[r][f"{arm}_AP_pct"]) for arm in arms}
    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    rc = {
        "font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "rev44-fresh-v1", "path.simplify": False, "axes.unicode_minus": True,
        "mathtext.fontset": "stixsans",
    }
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(rc):
        # Retain the original CSV reader, four raw series, marker selection and shading interval.
        fig = archive["make_figure"](rows)
        try:
            fig.set_size_inches(6.8, 6.8 * 2703443 / 6217920)
            fig.subplots_adjust(left=0.082, right=0.978, bottom=0.255, top=0.865, wspace=0.36)
            for legend in list(fig.legends):
                legend.remove()
            for ax, metric, title in zip(fig.axes, ("AP", "APs"), ("Validation AP", "Small-object validation AP")):
                ax.set_title("")
                ax.set_title(title, loc="left", fontsize=10.7, fontweight="bold", color=INK, pad=8)
                if metric == "APs":
                    ax.set_yticks([0, 3, 6, 9, 12, 15, 18])
                ax.set_xlabel("Completed training round", fontsize=8.6, color=INK, labelpad=4)
                ax.set_ylabel("AP (%)" if metric == "AP" else r"AP$_{s}$ (%)", fontsize=8.6, color=INK, labelpad=4)
                ax.tick_params(axis="both", labelsize=7.8, colors=MUTED, width=0.65, length=2.6)
                ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
                ax.set_axisbelow(True)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color("#879DA4")
                    ax.spines[side].set_linewidth(0.7)
                ax.patches[0].set_facecolor(SHADE)
                ax.patches[0].set_edgecolor("none")
                ax.patches[0].set_gid(f"{metric}-interval-297-320")
                for index, arm in enumerate(arms):
                    line = ax.lines[index]
                    line.set_linewidth(1.2)
                    line.set_gid(f"{metric}-{arm}-320-observations")
                    if len(line.get_xdata()) != 320:
                        raise ValueError("Incomplete archived validation curve")
                    # Both panels use the round selected by overall AP.
                    for offset, (tag, round_number) in enumerate((("end", 320), ("best-AP", peaks[arm]))):
                        marker = ax.collections[2 * index + offset]
                        if tuple(marker.get_offsets()[0]) != (round_number, rows[round_number][f"{arm}_{metric}_pct"]):
                            raise ValueError(f"Marker mismatch in {metric}, {arm}, round {round_number}")
                        marker.set_gid(f"{metric}-{arm}-{tag}-round-{round_number}")
                        marker.set_edgecolor("#273943")
                        marker.set_linewidth(0.45)
            handles = [Line2D([], [], color=colors[arm], linestyle=styles[arm], linewidth=1.4, label=arm)
                       for arm in arms]
            for tag, shape, size in (("Round 320", "o", 4.4), ("Best-AP round", "*", 7.2)):
                handles.append(Line2D([], [], marker=shape, markersize=size, linestyle="none",
                                      markeredgecolor=INK, markeredgewidth=0.5,
                                      markerfacecolor="#83979D", label=tag))
            handles.append(Patch(facecolor=SHADE, edgecolor="#A8BCBC", linewidth=0.5, label="Rounds 297–320"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.011),
                       ncol=5, frameon=False, fontsize=7.8, labelcolor=INK, borderpad=0.25,
                       handlelength=1.65, handletextpad=0.46, columnspacing=1.25)
            for text in fig.findobj(Text):
                text.set_fontfamily("Arial")
            info = {
                "Title": "FRESH validation AP and small-object AP over 320 training rounds",
                "Description": f"Unsmoothed CSV percentages. Stars use the highest overall AP (BASE round {peaks[arms[0]]}, FULL round {peaks[arms[1]]}) in both panels. Circles mark round 320. Shading covers integer rounds 297–320. Source CSV SHA-256: {source_hash}.",
            }
            fig.savefig(destination / f"{stem}.svg", format="svg", facecolor="white",
                        metadata={**info, "Date": None})
            fig.savefig(destination / f"{stem}.png", format="png", dpi=600, facecolor="white",
                        metadata=info)
        finally:
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCES,
                        help="Folder with the archived FRESH CSV and 对应代码/replot_fresh.py")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--stem", default="图6_FRESH验证曲线")
    options = parser.parse_args()
    draw(options.resources, options.output_dir, options.stem)

