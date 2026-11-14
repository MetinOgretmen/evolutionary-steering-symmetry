"""Separate a 35-degree clamp from a change in steering gain on frozen policies."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label

import pilot
from paper_corridors import Corridor, ROOT, SCALE, drive, load_corridors
from run_experiments import save_json
from source_reproduction import OPTIONS
from transfer_experiment import CONDITIONS, SEEDS
from verify_source_controller import interpolated_contacts


def audit(genome, course, network, gain, cap=None):
    log = []
    options = {**OPTIONS, "steering_scale": gain / 35, "steering_cap": cap}
    _, finish, progress, path = drive(genome, course=course, trace=True, steering_log=log, **network, **options)
    touches = None
    if finish[0]:
        touches = interpolated_contacts(genome, course, np.asarray(path), **network,
                                        steering_scale=options["steering_scale"], steering_cap=cap)
    angles = np.rad2deg(np.asarray([angle for angle, _ in log]))
    reason = "wall_or_obstacle_contact" if log[-1][1] else "timeout"
    if finish[0]:
        reason = "intermediate_contact" if touches else "finished"
    return {"reason": reason, "progress": float(progress[0]), "ticks": len(log),
            "peak_steering_deg": float(np.max(np.abs(angles))), "last_steering_deg": float(angles[-1]),
            "steps_above_35_deg": int(np.count_nonzero(np.abs(angles) > 35)),
            "last_position_m": path[-1].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure15-image", type=Path, help="Audit the square-closing mask from the embedded Figure 15 PNG")
    args = parser.parse_args()
    directory = ROOT / "results" / "transfer"
    network = CONDITIONS["symmetric"].network()
    courses = load_corridors()
    if args.figure15_image:
        audit_figure15_mask(directory, courses["map9"], args.figure15_image)
        return
    rows = []
    for seed in SEEDS:
        label = f"symmetric_restarts_{seed}"
        archive = json.loads((directory / f"weights_{label}.json").read_text())
        original = json.loads((directory / f"outcome_{label}.json").read_text())
        genome = np.asarray([archive["champion"]])
        for name, course in courses.items():
            baseline = audit(genome, course, network, 90)
            assert (baseline["reason"] == "finished") == original["courses"][name]["original"]["finished"]
            # If no command exceeded the cap, applying it cannot change the trajectory.
            capped = (audit(genome, course, network, 90, cap=np.deg2rad(35))
                      if baseline["steps_above_35_deg"] else baseline)
            reduced = audit(genome, course, network, 35)
            rows.append({"seed": seed, "course": name, "gain_90": baseline,
                         "gain_90_cap_35": capped, "gain_35": reduced})
        print(f"Checked {seed}", flush=True)
    save_json(directory / "steering_audit.json", rows)
    figure15 = [row for row in rows if row["course"] == "map9"]
    plot_figure15(directory, courses["map9"], figure15, network)


def alternate_figure15(course, image):
    rgb = np.asarray(Image.open(image).convert("RGB"))
    if rgb.shape[:2] != course.mask.shape:
        raise ValueError("Use the original embedded Figure 15 raster at 390 by 499 pixels.")
    barrier = np.pad(((rgb == [0, 0, 255]).all(axis=-1)
                      | (rgb == [0, 0, 127]).all(axis=-1)
                      | (rgb == [0, 0, 0]).all(axis=-1)), 2)
    closed = binary_closing(barrier, structure=np.ones((3, 3), dtype=bool))
    regions, _ = label(binary_fill_holes(closed) & ~closed)
    start_x, start_y = course.start_pixel
    goal_x, goal_y = course.goal_pixel
    region = regions[start_y + 2, start_x + 2]
    if region in (0, regions[0, 0]) or regions[goal_y + 2, goal_x + 2] != region:
        raise ValueError("The alternative mask does not contain an enclosed start-to-goal corridor.")
    mask = (regions == region)[2:-2, 2:-2]
    return Corridor(mask, course.start_pixel, course.goal_pixel)


def audit_figure15_mask(directory, course, image):
    alternative = alternate_figure15(course, image)
    results = {}
    for architecture, condition in CONDITIONS.items():
        rows = []
        for seed in SEEDS:
            run = f"{architecture}_restarts_{seed}"
            archive = json.loads((directory / f"weights_{run}.json").read_text())
            outcome = json.loads((directory / f"outcome_{run}.json").read_text())
            changed = audit(np.asarray([archive["champion"]]), alternative, condition.network(), 90)
            original = outcome["courses"]
            rows.append({"seed": seed, "baseline_finish": original["map9"]["original"]["finished"],
                         "finished_other_two": all(original[name]["original"]["finished"]
                                                   for name in ("map7", "map8")),
                         "square_closing": changed})
        results[architecture] = rows
    save_json(directory / "figure15_geometry_audit.json", {
        "original_free_pixels": int(course.mask.sum()), "alternative_free_pixels": int(alternative.mask.sum()),
        "results": results,
    })


def plot_figure15(directory, course, rows, network):
    fig, ax = pilot.plt.subplots(figsize=(6.5, 7.5), layout="constrained")
    height, width = course.mask.shape
    ax.imshow(course.mask, extent=(-0.5 * SCALE, (width - 0.5) * SCALE,
                                  -(height - 0.5) * SCALE, 0.5 * SCALE),
              cmap="Greys_r", alpha=0.32, interpolation="nearest")
    finishes = [row for row in rows if row["gain_90"]["reason"] == "finished"]
    for row in finishes:
        weights = json.loads((directory / f"weights_symmetric_restarts_{row['seed']}.json").read_text())
        _, _, _, path = drive(np.asarray([weights["champion"]]), course=course, trace=True,
                               **network, **OPTIONS)
        ax.plot(*np.asarray(path).T, lw=1.15, label=f"Finish · seed {row['seed']}")
    collisions = [row["gain_90"]["last_position_m"] for row in rows
                  if row["gain_90"]["reason"] == "wall_or_obstacle_contact"]
    ax.scatter(*np.asarray(collisions).T, color="#a02d29", marker="x", s=46, zorder=4,
               label="Vehicle reference at contact")
    ax.annotate("5 contacts near 57% progress", xy=(17, -164), xytext=(33, -139),
                arrowprops={"arrowstyle": "->", "color": "#a02d29"}, fontsize=9)
    ax.scatter(*course.start, color="#222222", s=22, zorder=5, label="Start")
    ax.scatter(*course.goal, facecolor="white", edgecolor="#222222", s=28, zorder=5, label="Finish area")
    ax.set(title="Figure 15 · 12 frozen symmetric controllers", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(directory / "figure15_contacts.png", dpi=180)
    pilot.plt.close(fig)


if __name__ == "__main__":
    main()
