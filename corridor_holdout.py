"""Replay the frozen corridor controller on Figure 17, unused in development."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, label

import pilot
from paper_corridors import Corridor, ROOT, SCALE, drive, load_corridors, reference


MASK = ROOT / "paper_figure17_mask.npz"
START = (937, 261)
GOAL = (37, 14)


def digitize(pdf):
    from pypdf import PdfReader

    image = next(image for image in PdfReader(pdf).pages[5].images if image.name == "Im17.png")
    rgb = np.asarray(image.image.convert("RGB"))
    barrier = ((rgb == [0, 0, 255]).all(axis=-1)
               | (rgb == [0, 0, 127]).all(axis=-1)
               | (rgb == [0, 0, 0]).all(axis=-1))
    barrier = np.pad(barrier, 2)
    regions, _ = label(binary_fill_holes(barrier) & ~barrier)
    region = regions[START[1] + 2, START[0] + 2]
    if not region or regions[GOAL[1] + 2, GOAL[0] + 2] != region:
        raise ValueError("Could not recover a connected Figure 17 corridor.")
    np.savez_compressed(MASK, free=(regions == region)[2:-2, 2:-2])


def sensor_check(genome, heldout, results):
    courses = {**load_corridors(), "figure17": heldout}
    rows = []
    for index, (name, course) in enumerate(courses.items()):
        conditions = [(25, 0.0, 1), (7, 0.0, 1), (25, 0.05, 6), (25, 0.10, 6)]
        for beams, sigma, repeats in conditions:
            for repeat in range(repeats):
                noise = None
                if sigma:
                    rng = np.random.default_rng([7250, index, repeat])
                    noise = sigma * rng.standard_normal((course.max_ticks, len(pilot.BEAM_ANGLES)))
                _, finished, progress, _ = drive(
                    genome, True, course, steering_scale=90 / 35, input_scale=pilot.SENSOR_RANGE / SCALE,
                    physical_beams=beams, noise=noise,
                )
                rows.append({"course": name, "physical_beams": beams, "noise_std": sigma,
                             "repeat": repeat, "finished": bool(finished[0]), "progress": float(progress[0])})
            count = sum(row["finished"] for row in rows[-repeats:])
            print(f"{name}, {beams} beams, noise {sigma:.0%}: {count}/{repeats} finished", flush=True)
    (results / "corridor_sensor_check.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digitize", type=Path)
    parser.add_argument("--sensor-check", action="store_true")
    args = parser.parse_args()
    if args.digitize:
        digitize(args.digitize)
    with np.load(MASK) as archive:
        course = Corridor(archive["free"], START, GOAL)
    can_finish, reference_path = reference(course, 5.0)
    if not can_finish:
        raise ValueError("Figure 17 could not be verified by the map-aware reference.")

    results = ROOT / "results"
    trial = json.loads((results / "corridor_development.json").read_text())
    genome = np.array([trial["champion"]])
    _, finished, progress, path = drive(genome, True, course, trace=True,
                                        steering_scale=90 / 35, input_scale=pilot.SENSOR_RANGE / SCALE)
    result = {
        "source_figure": 17,
        "model": "corridor_development.json",
        "role": "held_out",
        "finished": bool(finished[0]),
        "progress": float(progress[0]),
        "ticks": len(path) - 1,
        "reference_ticks": len(reference_path) - 1,
    }
    (results / "corridor_figure17_holdout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)

    fig, axis = pilot.plt.subplots(figsize=(10, 5.4), layout="constrained")
    height, width = course.mask.shape
    axis.imshow(course.mask, extent=(-0.5 * SCALE, (width - 0.5) * SCALE,
                                     -(height - 0.5) * SCALE, 0.5 * SCALE),
                cmap="Greys_r", alpha=0.32, interpolation="nearest")
    axis.plot(reference_path[:, 0], reference_path[:, 1], color="#266b88", label="Map-aware reference")
    path = np.asarray(path)
    axis.plot(path[:, 0], path[:, 1], color="#ba4e22", label="Frozen symmetric controller")
    axis.scatter(*course.start, color="#19804a", label="Start")
    axis.scatter(*course.goal, color="#bc4b36", label="Goal")
    axis.scatter(*path[-1], color="#ba4e22", marker="o" if finished[0] else "x", zorder=3)
    outcome = "Finished" if finished[0] else "Stopped before finish"
    axis.set(title=f"Held-out reconstruction of Figure 17 · {outcome}", xlabel="x [m]", ylabel="y [m]")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=4, fontsize=8)
    fig.savefig(results / "corridor_figure17_holdout.png", dpi=170)
    pilot.plt.close(fig)
    if args.sensor_check:
        sensor_check(genome, course, results)


if __name__ == "__main__":
    main()
