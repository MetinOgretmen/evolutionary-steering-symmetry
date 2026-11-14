"""Test the frozen guard and controllers on three additional corridor layouts."""

from concurrent.futures import ProcessPoolExecutor
import argparse
from hashlib import sha256

import numpy as np

import pilot
import footprint_guard as guard
from corridor_holdout import GOAL, MASK, START
from paper_corridors import Corridor, ROOT, SCALE, move_vehicle
from run_experiments import save_json
from transfer_experiment import SEEDS, guide_targets


LAYOUTS = {
    "switchbacks": (11, [(12, 12), (78, 12), (78, 47), (23, 47), (23, 88), (90, 88)]),
    "offset_bends": (10, [(15, 15), (15, 62), (60, 62), (60, 112), (114, 112), (114, 60), (162, 60)]),
    "zigzag": (12, [(14, 18), (68, 18), (68, 54), (114, 54), (114, 18), (170, 18), (170, 83), (117, 83)]),
}
NEW_TRACK = {
    "long_linked_bends": (8.5, [(148, 18), (110, 18), (110, 41), (151, 41), (151, 78), (90, 78),
                              (90, 117), (54, 117), (54, 92), (19, 92), (19, 49), (62, 49),
                              (62, 17), (24, 17)]),
}
ARCHIVES = {"symmetric": "transfer_verified_selection", "free": "transfer_verified_selection_free"}


def make_course(width, waypoints, entry_extension=0):
    route = np.asarray(waypoints, dtype=float)
    x, y = np.meshgrid(np.arange(int((max(route[:, 0]) + width) / SCALE)) * SCALE,
                       np.arange(int((max(route[:, 1]) + width) / SCALE)) * SCALE)
    mask = np.zeros_like(x, dtype=bool)
    radius = width / 2
    tangents = np.diff(route, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    segment_starts = route[:-1].copy()
    segment_starts[0] -= entry_extension * tangents[0]
    for start, end in zip(segment_starts, route[1:]):
        lower, upper = np.minimum(start, end) - radius, np.maximum(start, end) + radius
        mask |= (lower[0] <= x) & (x <= upper[0]) & (lower[1] <= y) & (y <= upper[1])
    # Put a wall protrusion before each second turn, on the inside of the bend.
    for index in range(0, len(tangents) - 1, 2):
        forward = tangents[index]
        side = tangents[index + 1]
        center = route[index + 1] - (8 + index % 3) * forward
        longitudinal = (x - center[0]) * forward[0] + (y - center[1]) * forward[1]
        lateral = (x - center[0]) * side[0] + (y - center[1]) * side[1]
        mask &= ~((np.abs(longitudinal) <= 1.25) & (lateral >= radius - 1.5) & (lateral <= radius))
    start, goal = (tuple(np.rint(point / SCALE).astype(int)) for point in route[[0, -1]])
    return Corridor(mask, start, goal)


def feasibility(course):
    """Use the map only to check that the fixed test geometry is traversable."""
    targets = guide_targets(course)
    position = course.start[None].copy()
    heading = np.array([course.heading])
    for tick in range(1, course.max_ticks + 1):
        x, y = np.rint(position[0] / [SCALE, -SCALE]).astype(int)
        dx, dy = targets[y, x] - position[0]
        steering = np.arctan2(2 * pilot.WHEELBASE * (np.cos(heading) * dy - np.sin(heading) * dx),
                              dx**2 + dy**2)
        end, orientation = move_vehicle(position, heading, steering, guard.VEHICLE.step_distance)
        bodies, angles = guard.between(position, heading, end, orientation, guard.PHASES)
        if pilot.collides(bodies[0], angles[0], course).any():
            raise ValueError(f"The independent map-aware feasibility driver hit a wall at tick {tick}.")
        position, heading = end, orientation
        if np.linalg.norm(position[0] - course.goal) <= 3:
            return tick
    raise ValueError("The independent feasibility driver timed out.")


def evaluate(architecture, seed, courses, recover=False, goal_assist=False):
    genome = guard.weights(seed, architecture, ARCHIVES[architecture])
    result = {"architecture": architecture, "seed": seed, "tests": {}}
    for name, course in courses.items():
        modes = {}
        for enabled in (False, True):
            trial = guard.rollout(genome, course, guarded=enabled, symmetric=architecture == "symmetric",
                                  recover=recover, goal_assist=enabled and goal_assist)
            modes["on" if enabled else "off"] = {key: trial[key]
                                                  for key in ("reason", "ticks", "progress", "interventions",
                                                              "short_steps", "goal_steps")}
        result["tests"][name] = modes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument("--stress", action="store_true",
                              help="Use narrow layouts and the separate Figure 17 corridor")
    layout_group.add_argument("--new-track", action="store_true",
                              help="Replay the final goal-aware controller on a separate fixed corridor")
    parser.add_argument("--recovery", action="store_true", help="Evaluate the experimental short-step guard")
    parser.add_argument("--goal-assist", action="store_true", help="Provide a goal cue to the guarded condition")
    args = parser.parse_args()
    if args.recovery and not args.stress:
        parser.error("--recovery requires --stress")
    if args.goal_assist and not args.recovery:
        parser.error("--goal-assist requires --recovery")
    recover = args.recovery or args.new_track
    goal_assist = args.goal_assist or args.new_track
    name = "fresh_guard_stress" if args.stress else "fresh_guard_check"
    if args.recovery:
        name += "_recovery"
    if args.goal_assist:
        name += "_goal_assist"
    if args.new_track:
        name = "final_guard_new_track"
    directory = ROOT / "results" / name
    directory.mkdir(exist_ok=True)
    layouts = LAYOUTS
    if args.stress:
        layouts = {label: (width - 3, waypoints) for label, (width, waypoints) in LAYOUTS.items()}
    if args.new_track:
        layouts = NEW_TRACK
    entry_extension = 2 if args.stress or args.new_track else 0
    courses = {name: make_course(*parameters, entry_extension=entry_extension)
               for name, parameters in layouts.items()}
    if args.stress:
        with np.load(MASK) as archive:
            courses["figure17"] = Corridor(archive["free"], START, GOAL)
    feasible = {name: feasibility(course) for name, course in courses.items()}
    manifest = {
        "layouts": layouts, "reference_finish_ticks": feasible,
        "seeds": SEEDS, "archives": ARCHIVES, "recovery": recover, "goal_assist": goal_assist,
        "entry_extension_m": entry_extension,
        "course_mask_sha256": {name: sha256(course.mask.tobytes()).hexdigest()
                               for name, course in courses.items()},
        "guard_sha256": sha256((ROOT / "footprint_guard.py").read_bytes()).hexdigest(),
        "weight_sha256": {architecture: {
            str(seed): sha256(guard.weights(seed, architecture, experiment).tobytes()).hexdigest() for seed in SEEDS
        } for architecture, experiment in ARCHIVES.items()},
    }
    save_json(directory / "manifest.json", manifest)
    print("Geometry and frozen configuration recorded", feasible, flush=True)
    jobs = [(architecture, seed) for architecture in ARCHIVES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(evaluate, architecture, seed, courses, recover, goal_assist)
                   for architecture, seed in jobs]
        results = [future.result() for future in pending]
    save_json(directory / "outcomes.json", results)
    summary = {}
    for architecture in ARCHIVES:
        rows = [row for row in results if row["architecture"] == architecture]
        summary[architecture] = {}
        for mode in ("off", "on"):
            finishes = [[row["tests"][name][mode]["reason"] == "finished" for name in courses] for row in rows]
            summary[architecture][mode] = {
                "by_track": dict(zip(courses, np.sum(finishes, axis=0).tolist())),
                "all_layouts": [row["seed"] for row, completed in zip(rows, finishes) if all(completed)],
                "body_contacts": sum(test[mode]["reason"] == "body_contact" for row in rows
                                     for test in row["tests"].values()),
                "guard_stops": sum(test[mode]["reason"] == "stopped_by_guard" for row in rows
                                   for test in row["tests"].values()),
            }
    save_json(directory / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
