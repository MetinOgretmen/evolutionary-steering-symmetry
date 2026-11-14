"""Plot course trajectories, initialization sweeps and geometry benchmark results."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import pilot
from paper_corridors import ROOT


def read_table(path):
    with (ROOT / "results" / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_courses(output):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    from footprint_guard import rollout, weights
    from paper_corridors import load_corridors

    goal_radius = 3.0
    driver = weights(7316, "symmetric", "transfer_verified_selection")
    courses = load_corridors()
    fig, axes = pilot.plt.subplots(1, 4, figsize=(10, 3.31))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.035, top=0.895, wspace=0.12)
    for ax, title, course in zip(
        axes, ("T · training", "A · test", "B · test", "C · test"), courses.values(), strict=True
    ):
        run = rollout(driver, course)
        if run["reason"] != "finished":
            raise ValueError(f"Archived run 7316 did not finish {title}.")
        walls = course.boxes.reshape(-1, 2, 2)
        ax.add_collection(LineCollection(walls, colors="#748491", linewidths=0.65))
        ax.plot(*np.asarray(run["path"]).T, color="#146681", linewidth=1.1)
        ax.plot(*course.start, "o", color="#146681", markersize=5)
        ax.add_patch(Circle(course.goal, goal_radius, fill=False, edgecolor="#bf382e",
                            linewidth=1.25, zorder=5))
        wall_points = walls.reshape(-1, 2)
        lower = np.minimum(wall_points.min(axis=0), course.goal - goal_radius) - 3
        upper = np.maximum(wall_points.max(axis=0), course.goal + goal_radius) + 3
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_axis_off()
    fig.savefig(output, dpi=200)
    pilot.plt.close(fig)


def plot_scale(output):
    rows = read_table("followup_initial/scale_sweep.csv")
    fig, ax = pilot.plt.subplots(figsize=(6.5, 2.35), layout="constrained")
    for index, (key, label, colour, marker) in enumerate((
        ("symmetric", "Paired-input symmetry", "#176983", "o"),
        ("free", "Unrestricted", "#9c4932", "s"),
        ("free_small", "Smaller unrestricted", "#63666a", "x"),
        ("projected_context", "Output symmetry", "#654b9a", "^"),
    )):
        arm = [row for row in rows if row["architecture"] == key]
        scales = np.array([float(row["scale"]) for row in arm])
        counts = np.array([int(row["finishes"]) for row in arm])
        limits = np.array([binomtest(count, int(row["candidates"])).proportion_ci()
                           for row, count in zip(arm, counts, strict=True)]) * 2400
        ax.plot(scales, counts, label=label, color=colour, marker=marker, linewidth=1.3, markersize=4)
        ax.vlines(scales * 2 ** ((index - 1.5) * 0.075), limits[:, 0], limits[:, 1],
                  color=colour, alpha=0.55, linewidth=0.9)
        if key == "free":
            for scale, count in zip(scales, counts, strict=True):
                if count:
                    ax.annotate(str(count), (scale, count), xytext=(4, 7),
                                textcoords="offset points", color=colour, fontsize=8)
    scales = sorted({float(row["scale"]) for row in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales, labels=[f"{scale:g}" for scale in scales])
    ax.set(xlabel="Initial weight multiplier", ylabel="Finishes / 2,400", ylim=(-3, 70))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8, ncol=4)
    fig.savefig(output, dpi=250)
    pilot.plt.close(fig)


def plot_geometry(output):
    cells = defaultdict(lambda: defaultdict(list))
    for method, path in (("ga", "followup_courses/drives.csv"), ("cma", "followup_cma/procedural_drives.csv")):
        for row in read_table(path):
            if row["arm"] in ("symmetric", "free"):
                key = (float(row["width_m"]), float(row["depth_m"]), f"{method}_{row['arm']}")
                cells[key][int(row["seed"])].append(row["finish"] == "True")
    widths = sorted({width for width, _, _ in cells})
    draw = np.random.default_rng(4350).integers(0, 12, size=(20000, 12))
    fig, axes = pilot.plt.subplots(1, len(widths), figsize=(7.15, 2.2), sharey=True, layout="constrained")
    for ax, width in zip(axes, widths, strict=True):
        for index, (arm, colour, style, marker, label) in enumerate((
            ("ga_symmetric", "#176983", "-", "o", "GA symmetric"),
            ("ga_free", "#9c4932", "-", "s", "GA unrestricted"),
            ("cma_symmetric", "#176983", "--", "^", "CMA-ES symmetric"),
            ("cma_free", "#9c4932", "--", "x", "CMA-ES unrestricted"),
        )):
            depths = sorted(depth for w, depth, key in cells if w == width and key == arm)
            rates = []
            limits = []
            for depth in depths:
                by_seed = cells[width, depth, arm]
                finish_rate = np.array([np.mean(by_seed[seed]) for seed in sorted(by_seed)])
                assert len(finish_rate) == 12
                rates.append(finish_rate.mean())
                limits.append(np.quantile(finish_rate[draw].mean(axis=1), [0.025, 0.975]))
            ax.plot(depths, rates, color=colour, linestyle=style if width > 4.5 else "None",
                    marker=marker, linewidth=1.1, markersize=3, label=label)
            if width > 4.5:
                bounds = np.asarray(limits)
                ax.vlines(np.array(depths) + (index - 1.5) * 0.045, bounds[:, 0], bounds[:, 1],
                          color=colour, alpha=0.5, linewidth=0.9)
        title = f"Width {width:g} m" if width > 4.5 else "4.5 m (descriptive)"
        ax.set(title=title, ylim=(-0.04, 1.04), xlim=(-0.16, 2.41),
               xticks=[0, 0.75, 1.5, 2.25], xticklabels=["0", "0.75", "1.5", "2.25"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Finish fraction")
    fig.supxlabel("Inside protrusion depth (m)", fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=7)
    fig.savefig(output, dpi=250)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--courses-only", action="store_true", help="Regenerate the course overview only")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.courses_only:
        plot_courses(args.destination / "courses.png")
        return
    plot_scale(args.destination / "scale_sweep.png")
    plot_geometry(args.destination / "transfer_by_geometry.png")


if __name__ == "__main__":
    main()
