"""Evaluate frozen controllers on an independently generated, fixed corner grid."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

import pilot
from audit_course_clearance import clearance, sampled_body_poses
from followup_initial import write_csv
from followup_mutation import DESTINATION as MUTATION_RESULTS, seed_interval
from footprint_guard import rollout, weights
from fresh_guard_check import ARCHIVES, feasibility
from paper_corridors import Corridor, ROOT, SCALE
from run_experiments import save_json
from transfer_experiment import SEEDS

DESTINATION = ROOT / "results" / "followup_courses"
WIDTHS = (4.5, 6.0, 7.5, 9.0)
DEPTHS = (0.0, 0.75, 1.5, 2.25)
ARMS = ("symmetric", "free", "free_mutation_052")
LAYOUT_SEED = 435071
REFLECTION_AXIS = 40.0


def inside(x, y, rectangles):
    return np.logical_or.reduce([(x0 <= x) & (x < x1) & (y0 <= y) & (y < y1)
                                for x0, y0, x1, y1 in rectangles])


def exact_walls(free, excluded):
    """Find the boundary of a rectangle union minus a rectangular protrusion."""
    rectangles = free + excluded
    xs = np.unique([r[i] for r in rectangles for i in (0, 2)])
    ys = np.unique([r[i] for r in rectangles for i in (1, 3)])
    x, y = np.meshgrid((xs[1:] + xs[:-1]) / 2, (ys[1:] + ys[:-1]) / 2)
    active = inside(x, y, free) & ~inside(x, y, excluded) if excluded else inside(x, y, free)
    vertical = np.diff(np.pad(active, ((0, 0), (1, 1))), axis=1)
    horizontal = np.diff(np.pad(active, ((1, 1), (0, 0))), axis=0)
    return np.array([[xs[j], -ys[i + 1], xs[j], -ys[i]] for i, j in np.argwhere(vertical)]
                    + [[xs[j], -ys[i], xs[j + 1], -ys[i]] for i, j in np.argwhere(horizontal)])


def make_corner(width, depth, entry, exit_length, setback, direction):
    radius = width / 2
    sx, sy = 12.0, REFLECTION_AXIS
    turn = sx + entry
    free = [(sx - 7, sy - radius, turn + radius, sy + radius),
            (turn - radius, sy, turn + radius, sy + exit_length + 5)]
    excluded = [(turn - setback - 1.25, sy + radius - depth,
                 turn - setback + 1.25, sy + radius)] if depth else []
    goal_y = sy + exit_length
    if direction == "left":
        def mirror(rectangle):
            x0, y0, x1, y1 = rectangle
            return x0, 2 * sy - y1, x1, 2 * sy - y0
        free, excluded = [mirror(r) for r in free], [mirror(r) for r in excluded]
        goal_y = sy - exit_length
    x, y = np.meshgrid(np.arange(0, turn + radius + 3, SCALE),
                       np.arange(0, 2 * sy + SCALE, SCALE))
    # A raster point is inside only if all four nearby quadrants are free.
    epsilon = 1e-6
    mask = np.logical_and.reduce([inside(x + dx, y + dy, free)
                                  for dx in (-epsilon, epsilon) for dy in (-epsilon, epsilon)])
    if excluded:
        x0, y0, x1, y1 = excluded[0]
        mask &= ~((x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1))
    start = tuple(np.rint(np.array([sx, sy]) / SCALE).astype(int))
    goal = tuple(np.rint(np.array([turn, goal_y]) / SCALE).astype(int))
    course = Corridor(mask, start, goal)
    course.heading = 0.0
    course.boxes = exact_walls(free, excluded)
    course.centers = (course.boxes[:, :2] + course.boxes[:, 2:]) / 2
    course.halves = (course.boxes[:, 2:] - course.boxes[:, :2]) / 2
    course.vertical = pilot.axis_segments(course.boxes, 0)
    course.horizontal = pilot.axis_segments(course.boxes, 1)
    return course


def layouts():
    rng = np.random.default_rng(LAYOUT_SEED)
    variations = np.column_stack((rng.choice(np.arange(22, 33), 6, replace=False),
                                  rng.choice(np.arange(18, 27), 6, replace=False),
                                  rng.choice(np.arange(7, 15), 6, replace=False)))
    return [{"course": f"w{wi}_d{di}_l{i}_{direction}", "width_m": width, "depth_m": depth,
             "layout": i, "entry_m": int(entry), "exit_m": int(exit_length), "setback_m": int(setback),
             "turn": direction}
            for wi, width in enumerate(WIDTHS) for di, depth in enumerate(DEPTHS)
            for i, (entry, exit_length, setback) in enumerate(variations)
            for direction in ("right", "left")]


def build(spec):
    return make_corner(spec["width_m"], spec["depth_m"], spec["entry_m"], spec["exit_m"],
                       spec["setback_m"], spec["turn"])


def reference_result(spec):
    course = build(spec)
    try:
        return {"course": spec["course"], "reference_ticks": feasibility(course), "failure": ""}
    except ValueError as error:
        return {"course": spec["course"], "reference_ticks": None, "failure": str(error)}


def replay(spec, drivers):
    course = build(spec)
    rows = []
    for arm, seed, genome in drivers:
        run = rollout(genome, course, symmetric=arm == "symmetric")
        assert run["interventions"] == 0
        bodies, headings = sampled_body_poses(run, course)
        measure = clearance(bodies, headings, course)
        assert (measure["first_contact_step"] is not None) == (run["reason"] == "body_contact")
        rows.append({**spec, "arm": arm, "seed": seed, "finish": run["reason"] == "finished",
                     "reason": run["reason"], "steps": run["ticks"], **measure})
    return rows


def summarize(rows):
    by_geometry = []
    by_driver = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for arm in ARMS:
                cell = [r for r in rows if (r["width_m"], r["depth_m"], r["arm"]) == (width, depth, arm)]
                successful = [r["minimum_sampled_clearance_m"] for r in cell if r["finish"]]
                if cell:
                    by_geometry.append({"width_m": width, "depth_m": depth, "arm": arm,
                                        "courses": len({r["course"] for r in cell}), "drives": len(cell),
                                        "finishes": len(successful), "finish_fraction": len(successful) / len(cell),
                                        "median_finish_clearance_m": float(np.median(successful)) if successful else ""})
    for arm in ARMS:
        for seed in SEEDS:
            subset = [r for r in rows if (r["arm"], r["seed"]) == (arm, seed)]
            finished = [r for r in subset if r["finish"]]
            by_driver.append({"arm": arm, "seed": seed, "courses": len(subset), "finishes": len(finished),
                              "finish_fraction": len(finished) / len(subset)})
    rates = {(r["arm"], r["seed"]): r["finish_fraction"] for r in by_driver}
    comparisons = {}
    for arm, control in (("symmetric", "free"), ("symmetric", "free_mutation_052"),
                         ("free_mutation_052", "free")):
        comparisons[f"{arm}_minus_{control}"] = seed_interval([rates[arm, seed] - rates[control, seed] for seed in SEEDS])
    return by_geometry, by_driver, comparisons


def plot(by_geometry, output):
    fig, axes = pilot.plt.subplots(1, len(WIDTHS), figsize=(7.1, 1.95), sharex=True, sharey=True, layout="constrained")
    for ax, width in zip(axes, WIDTHS, strict=True):
        for arm, color, label, marker in zip(ARMS, ("#176983", "#9c4932", "#654b9a"),
                                            ("Symmetric", "Free, 10%", "Free, 5.2%"), ("o", "s", "^")):
            lookup = {r["depth_m"]: r["finish_fraction"] for r in by_geometry
                      if r["arm"] == arm and r["width_m"] == width}
            ax.plot(DEPTHS, [lookup.get(depth, np.nan) for depth in DEPTHS],
                    marker=marker, markersize=3, linewidth=1.2, color=color, label=label)
        ax.set(title=f"Width {width:g} m", ylim=(-0.05, 1.05), xticks=DEPTHS)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Finish fraction")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.supxlabel("Inside protrusion depth (m)", fontsize=9)
    fig.savefig(output, dpi=250)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    DESTINATION.mkdir(exist_ok=True)
    specs = layouts()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        checks = list(pool.map(reference_result, specs, chunksize=3))
        checked = {r["course"]: r for r in checks}
        eligible = {spec["course"] for spec in specs if checked[spec["course"]]["reference_ticks"] is not None}
        for a, b in zip(specs[::2], specs[1::2], strict=True):
            if a["course"] not in eligible or b["course"] not in eligible:
                eligible.difference_update((a["course"], b["course"]))
        write_csv(DESTINATION / "course_manifest.csv",
                  [{**spec, **checked[spec["course"]], "eligible": spec["course"] in eligible} for spec in specs])
        print(f"{len(eligible)}/{len(specs)} courses passed the paired feasibility check", flush=True)
        drivers = [(arm, seed, weights(seed, arm, archive)) for arm, archive in ARCHIVES.items() for seed in SEEDS]
        drivers += [("free_mutation_052", seed,
                     np.asarray([json.loads((MUTATION_RESULTS / f"run_{seed}.json").read_text())["champion"]]))
                    for seed in SEEDS]
        jobs = [pool.submit(replay, spec, drivers) for spec in specs if spec["course"] in eligible]
        rows = []
        for index, job in enumerate(jobs, 1):
            rows.extend(job.result())
            if index % 24 == 0:
                print(f"Replayed {index}/{len(jobs)} courses", flush=True)
    write_csv(DESTINATION / "drives.csv", rows)
    geometry, by_driver, comparisons = summarize(rows)
    write_csv(DESTINATION / "geometry_summary.csv", geometry)
    write_csv(DESTINATION / "driver_summary.csv", by_driver)
    save_json(DESTINATION / "summary.json", {"generated": len(specs), "eligible": len(eligible),
                                            "comparisons": comparisons})
    plot(geometry, DESTINATION / "transfer_by_geometry.png")


if __name__ == "__main__":
    main()
