"""Check the controller identities and the geometry used in the follow-up."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

import pilot
from analyze_steering_behavior import steer, track_scans
from followup_courses import DESTINATION, REFLECTION_AXIS, build, layouts
from followup_mutation import DESTINATION as MUTATION_RESULTS
from footprint_guard import weights
from fresh_guard_check import ARCHIVES
from paper_corridors import load_corridors
from transfer_experiment import CONDITIONS, SEEDS, assess, mirrored


def check_geometry():
    specs = layouts()
    for right_spec, left_spec in zip(specs[::2], specs[1::2], strict=True):
        right, left = build(right_spec), build(left_spec)
        np.testing.assert_array_equal(right.mask[::-1], left.mask)
        reflected = right.boxes.copy()
        reflected[:, [1, 3]] = -2 * REFLECTION_AXIS - reflected[:, [3, 1]]
        assert sorted(map(tuple, reflected)) == sorted(map(tuple, left.boxes))
        headings = np.array([0.0, -0.17, 0.28])
        poses = np.array([right.start, right.start + [6, 0.25], right.start + [9, -0.25]])
        opposite = poses.copy()
        opposite[:, 1] = -2 * REFLECTION_AXIS - opposite[:, 1]
        readings = pilot.sense(poses - pilot.HALF_LENGTH * pilot.direction(headings), headings, right)
        mirror = pilot.sense(opposite - pilot.HALF_LENGTH * pilot.direction(-headings), -headings, left)
        np.testing.assert_allclose(readings, mirror[:, ::-1], atol=1e-12, rtol=0)


def check_controller():
    course = load_corridors()["map6"]
    scans = track_scans(course)
    for architecture, genes in (("symmetric", 210), ("free", 405)):
        np.testing.assert_array_equal(steer(np.zeros((1, genes)), scans, CONDITIONS[architecture]), 0)
    symmetric = CONDITIONS["symmetric"]
    drivers = np.vstack([weights(seed, "symmetric", ARCHIVES["symmetric"]) for seed in SEEDS])
    np.testing.assert_allclose(steer(drivers, scans, symmetric),
                               -steer(drivers, scans[:, ::-1], symmetric), atol=1e-10, rtol=0)
    example = weights(7316, "symmetric", ARCHIVES["symmetric"])
    assert assess(example, course, symmetric)["finished"]
    assert assess(example, mirrored(course), symmetric)["finished"]


def check_outputs():
    with (DESTINATION / "course_manifest.csv").open() as stream:
        eligible = {r["course"] for r in csv.DictReader(stream) if r["eligible"] == "True"}
    with (DESTINATION / "drives.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    pairs = {(r["arm"], r["seed"], r["course"]): r for r in rows}
    assert len(rows) == len(pairs) == 3 * len(SEEDS) * len(eligible)
    assert {r["course"] for r in rows} == eligible
    for seed in SEEDS:
        run = json.loads((MUTATION_RESULTS / f"run_{seed}.json").read_text())
        assert run["mutation_probability_per_gene"] == 0.052
        assert [row["generation"] for row in run["fitness_history"]] == list(range(1, 26))
        assert all(row["best_fitness"] >= row["median_fitness"] for row in run["fitness_history"])
        assert len(run["champion"]) == 405
    mismatches = Counter()
    for (arm, seed, course), row in pairs.items():
        if course.endswith("_right"):
            mirror = pairs[arm, seed, course.removesuffix("_right") + "_left"]
            mismatches[arm] += row["finish"] != mirror["finish"]
    return {"eligible_courses": len(eligible), "complete_replays": len(rows),
            "mirror_finish_disagreements": mismatches}


def plot_first_finishes(output):
    rows = json.loads((MUTATION_RESULTS / "summary.json").read_text())["runs"]
    fig, ax = pilot.plt.subplots(figsize=(6.1, 2.15), layout="constrained")
    for arm, label, colour, style in (("symmetric", "Symmetric, 10%", "#176983", "-"),
                                       ("free", "Unrestricted, 10%", "#9c4932", "-"),
                                       ("free_mutation_052", "Unrestricted, 5.2%", "#654b9a", "--")):
        data = [r["first_verified_population"] for r in rows if r["arm"] == arm]
        x = np.arange(1, 10)
        ax.step(x, [sum(value is not None and value <= gen for value in data) for gen in x],
                where="post", linewidth=1.5, label=label, color=colour, linestyle=style)
    ax.set(xlabel="Generation (1 = initial random population)", ylabel="Runs with a driver",
           xticks=x, yticks=np.arange(0, 13, 2), ylim=(0, 12.5))
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(output, dpi=250)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    check_controller()
    check_geometry()
    print(json.dumps(check_outputs(), indent=2))
    if args.plot:
        plot_first_finishes(args.plot)


if __name__ == "__main__":
    main()
