"""Measure the sampled body clearance of frozen, unassisted course replays."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pilot
from audit_figure15_corner import footprint_margin
from audit_steering import alternate_figure15
from footprint_guard import PHASES, VEHICLE, between, rollout, weights
from fresh_guard_check import ARCHIVES
from paper_corridors import ROOT, load_corridors, move_vehicle
from run_experiments import save_json
from transfer_experiment import SEEDS


BODY_RADIUS = np.hypot(pilot.HALF_LENGTH, pilot.HALF_WIDTH)


def sampled_body_poses(run, course, phases=PHASES):
    path = np.asarray(run["path"])
    headings = [course.heading]
    for index, (_, command, _) in enumerate(run["commands"]):
        end, heading = move_vehicle(path[index:index + 1], np.asarray(headings[-1:]),
                                    np.deg2rad([command]), VEHICLE.step_distance)
        np.testing.assert_allclose(end[0], path[index + 1], atol=1e-9, rtol=0)
        headings.append(float(heading[0]))
    headings = np.asarray(headings)
    bodies, angles = between(path[:-1], headings[:-1], path[1:], headings[1:], phases)
    initial = path[0] + VEHICLE.body_offset * pilot.direction(headings[0])
    return np.vstack((initial, bodies.reshape(-1, 2))), np.r_[headings[0], angles.ravel()]


def clearance(bodies, headings, course, samples_per_step=len(PHASES)):
    walls = course.boxes.reshape(-1, 2, 2)
    minimum = np.inf
    for index, (body, heading) in enumerate(zip(bodies, headings)):
        gap = np.maximum(0, np.maximum(walls[:, 0] - body, body - walls[:, 1]))
        nearby = np.sum(gap ** 2, axis=1) <= (minimum + BODY_RADIUS + 1e-9) ** 2
        if not nearby.any():
            continue
        margin, _ = footprint_margin(body, heading, walls[nearby])
        minimum = min(minimum, margin)
        if margin <= 0:
            return {"minimum_sampled_clearance_m": minimum,
                    "first_contact_step": (index - 1) // samples_per_step + 1 if index else 0}
    return {"minimum_sampled_clearance_m": minimum, "first_contact_step": None}


def run_case(architecture, seed, name, courses):
    course = courses[name]
    genome = weights(seed, architecture, ARCHIVES[architecture])
    run = rollout(genome, course, symmetric=architecture == "symmetric")
    assert run["interventions"] == run["short_steps"] == run["goal_steps"] == 0
    bodies, headings = sampled_body_poses(run, course)
    measurement = clearance(bodies, headings, course)
    if run["reason"] == "body_contact":
        assert measurement["first_contact_step"] == run["ticks"]
        valid_poses = pilot.collides(bodies, headings, course)
        stop = np.flatnonzero(valid_poses)[0] + 1
    else:
        assert measurement["first_contact_step"] is None
        stop = len(bodies)
    fixed_path = {}
    if name in ("map9", "map9_alternate"):
        other = "map9" if name == "map9_alternate" else "map9_alternate"
        fixed_path[other] = clearance(bodies[:stop], headings[:stop], courses[other])
    return {"architecture": architecture, "seed": seed, "course": name,
            "reason": run["reason"], "steps": run["ticks"],
            "progress": run["progress"], **measurement, "fixed_path_comparison": fixed_path}


def refine_case(architecture, seed, name, courses):
    course = courses[name]
    genome = weights(seed, architecture, ARCHIVES[architecture])
    run = rollout(genome, course, symmetric=architecture == "symmetric")
    assert run["reason"] == "finished"
    phases = np.arange(1, 101) / 100
    bodies, headings = sampled_body_poses(run, course, phases)
    return {"architecture": architecture, "seed": seed, "course": name,
            **clearance(bodies, headings, course, len(phases))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure15_image", type=Path, help="Original embedded Figure 15 PNG")
    parser.add_argument("--refine-close-finishes", action="store_true",
                        help="Use 100 poses per step for close finishes on A, B and C")
    args = parser.parse_args()
    courses = load_corridors()
    courses.pop("map6")
    courses["map9_alternate"] = alternate_figure15(courses["map9"], args.figure15_image)
    destination = ROOT / "results" / "course_clearance"
    destination.mkdir(exist_ok=True)
    if args.refine_close_finishes:
        import json

        previous = json.loads((destination / "replays.json").read_text())["runs"]
        selected = [r for r in previous if r["course"] != "map9_alternate"
                    and r["reason"] == "finished" and r["minimum_sampled_clearance_m"] <= 0.10]
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(refine_case, r["architecture"], r["seed"], r["course"], courses)
                       for r in selected]
            rows = [{"ten_pose_clearance_m": r["minimum_sampled_clearance_m"], **job.result()}
                    for r, job in zip(selected, pending)]
        save_json(destination / "dense_close_finishes.json", {"samples_per_step": 100, "runs": rows})
        return
    cases = [(architecture, seed, name) for architecture in ARCHIVES for seed in SEEDS for name in courses]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(run_case, *case, courses) for case in cases]
        rows = [task.result() for task in pending]
    save_json(destination / "replays.json", {
        "sampling": "start and ten sampled poses per step; stop at first sampled contact",
        "cross_replay": "source poses up to first sampled source contact or source termination",
        "negative_clearance": "negative pairwise separating-axis margin at first contact, not penetration after collision",
        "runs": rows,
    })


if __name__ == "__main__":
    main()
