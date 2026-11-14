"""Compare restart and corner-focused neuroevolution on the same training geometry."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

import numpy as np

import pilot
from corridor_search import PreparedCourse
from footprint_guard import PHASES, between, rollout
from fresh_guard_check import feasibility, make_course
from paper_corridors import Corridor, MASKS, ROOT, SCALE, SPECS, move_vehicle
from run_experiments import save_json
from source_reproduction import OPTIONS
from transfer_experiment import CONDITIONS, GENERATIONS, POPULATION, SEEDS, guide_targets, restart_scenarios


POSE_SAMPLES = 10
VEHICLE = OPTIONS["vehicle"]
# Four turns ranked by change of route direction over 12 pixels each side,
# with at least 65 route samples between turns and two turns per direction.
CORNERS = (314, 871, 517, 1593)
HOLDOUTS = {
    "nested_turns": (9, [(18, 18), (61, 18), (61, 44), (104, 44), (104, 85),
                          (68, 85), (68, 110), (135, 110), (135, 63), (168, 63)]),
    "alternating_offsets": (9.5, [(175, 18), (175, 52), (135, 52), (135, 25), (91, 25),
                                   (91, 77), (52, 77), (52, 47), (16, 47), (16, 105), (69, 105)]),
    "long_return": (10, [(20, 20), (69, 20), (69, 47), (22, 47), (22, 91),
                          (79, 91), (79, 64), (131, 64), (131, 114), (170, 114), (170, 81)]),
}


@dataclass
class Trial:
    course: Corridor
    gate: np.ndarray | None = None
    remaining: float = 0.0


def route_distance(course, position):
    x, y = np.rint(position / [SCALE, -SCALE]).astype(int)
    return float(course.distance_to_goal[y, x])


def corner_trials(course):
    trials = []
    route = course.route
    for index in CORNERS:
        entry, exit_point = route[index - 22], route[index + 27]
        incoming = route[index - 17] - route[index - 27]
        heading = np.arctan2(incoming[1], incoming[0])
        normal = pilot.direction(heading + np.pi / 2)
        outgoing = route[index + 33] - route[index + 21]
        outgoing /= np.linalg.norm(outgoing)
        gate = np.r_[exit_point, outgoing, 3.0]
        remaining = route_distance(course, exit_point)
        for side in (-1, 1):
            scenario = copy(course)
            scenario.start = entry + side * 0.6 * normal
            scenario.heading = heading + side * np.deg2rad(10)
            scenario.length = route_distance(course, scenario.start) - remaining
            scenario.max_ticks = 60
            body = scenario.start + VEHICLE.body_offset * pilot.direction(scenario.heading)
            if not 0 < scenario.length < np.inf or pilot.collides(
                    body[None], np.array([scenario.heading]), course).any():
                raise ValueError(f"Invalid corner starting pose at {index}, side {side}")
            trials.append(Trial(scenario, gate, remaining))
    return trials


def corner_feasibility(trial, route):
    """Find a map-aware witness for geometry QA, separate from GA scoring."""
    course = trial.course
    gate_position, forward = trial.gate[:2], trial.gate[2:4]
    sideways = np.array([-forward[1], forward[0]])
    start_index = np.argmin(np.linalg.norm(route - course.start, axis=1))
    for preview in (3.5, 4.3, 5.2, 6.0, 7.0, 8.0):
        position = course.start[None].copy()
        heading = np.array([course.heading])
        nearest = start_index
        for tick in range(1, course.max_ticks + 1):
            window = route[nearest:nearest + 24]
            nearest += np.argmin(np.linalg.norm(window - position[0], axis=1))
            upcoming = route[nearest:]
            ahead = np.flatnonzero(np.linalg.norm(upcoming - position[0], axis=1) >= preview)
            target = upcoming[ahead[0]] if len(ahead) else upcoming[-1]
            dx, dy = target - position[0]
            steering = np.arctan2(2 * pilot.WHEELBASE * (np.cos(heading) * dy - np.sin(heading) * dx),
                                  dx**2 + dy**2)
            end, end_heading = move_vehicle(position, heading, steering, VEHICLE.step_distance)
            bodies, angles = between(position, heading, end, end_heading, PHASES)
            if pilot.collides(bodies[0], angles[0], course).any():
                break
            direction = pilot.direction(end_heading[0])
            relative = bodies[0, -1] - gate_position
            rear = (relative @ forward - pilot.HALF_LENGTH * abs(direction @ forward)
                   - pilot.HALF_WIDTH * abs(direction @ sideways))
            if rear >= 0 and abs(relative @ sideways) <= trial.gate[-1]:
                return tick
            position, heading = end, end_heading
    raise ValueError(f"No reference witness for corner starting at {course.start.tolist()}")


def score(prepared, genomes, trials, condition, focus, budget):
    scores = np.zeros(len(genomes))
    completed = np.zeros(len(genomes), dtype=int)
    for index, trial in enumerate(trials):
        progress, finished, ticks = prepared.evaluate(
            genomes, VEHICLE, scenario=trial.course, pose_samples=POSE_SAMPLES,
            exit_gate=trial.gate, exit_remaining=trial.remaining, **condition.network(),
        )
        budget["episodes"] += len(genomes)
        budget["ticks"] += int(ticks.sum())
        scores += progress ** 2 * trial.course.length / ticks
        if index == 0:
            nominal = finished
        else:
            completed += finished
    if focus:
        # At most one point of partial credit cannot outweigh one completed corner.
        scores = 10 * nominal + completed + scores / (1 + scores)
    return scores, nominal, completed


def train(course, architecture, seed, focus, generations, population):
    start_time = perf_counter()
    condition = CONDITIONS[architecture]
    rng = np.random.default_rng(seed)
    scenario_rng = np.random.default_rng([731, seed])
    ga = pilot.GASettings(population=population)
    genes = (14 if condition.symmetric else 27) * condition.hidden
    genomes = rng.uniform(-1, 1, (population, genes))
    prepared = PreparedCourse(course)
    nominal = Trial(course)
    extras = corner_trials(course) if focus else [Trial(scenario) for scenario in restart_scenarios(course)]
    shortlist = []
    training = {"episodes": 0, "ticks": 0}
    selection = {"episodes": 0, "ticks": 0}
    first_nominal = None
    first_nominal_and_both = None
    for generation in range(1, generations + 1):
        # Each population sees the full course plus two starts. In the corner arm
        # the first four trials are left turns and the other four right turns.
        trials = [nominal, extras[scenario_rng.integers(4)], extras[scenario_rng.integers(4, 8)]]
        scores, finished, local_finishes = score(prepared, genomes, trials, condition, focus, training)
        if first_nominal is None and finished.any():
            first_nominal = generation
        if first_nominal_and_both is None and np.any(finished & (local_finishes == 2)):
            first_nominal_and_both = generation
        top = np.argsort(-scores, kind="stable")[:3]
        shortlist.extend(genomes[top].copy())
        if generation < generations:
            genomes = pilot.offspring(genomes, scores, rng, ga)

    candidates = np.unique(np.asarray(shortlist), axis=0)
    scores, finished, local_finishes = score(prepared, candidates, [nominal, *extras], condition, focus, selection)
    rank = np.lexsort((-scores, -local_finishes, ~finished))
    winner = rank[0]
    verification = {"episodes": 0, "ticks": 0}
    verified_nominal = False
    for index in rank:
        replay = rollout(candidates[index:index + 1], course, symmetric=condition.symmetric)
        verification["episodes"] += 1
        verification["ticks"] += replay["ticks"]
        if replay["reason"] == "finished":
            winner = index
            verified_nominal = True
            break
    return {
        "architecture": architecture, "seed": seed, "corner_focus": focus, "genes": genes,
        "first_nominal_population_fast": first_nominal,
        "first_nominal_and_two_extras_population_fast": first_nominal_and_both,
        "training": {"candidate_slots": generations * population, **training},
        "selection": {"unique_candidates": len(candidates), **selection},
        "nominal_verification": verification,
        "selection_nominal_finish_verified": verified_nominal,
        "verification_changed_candidate": bool(winner != rank[0]),
        "selection_nominal_finish_fast": bool(finished[winner]),
        "selection_extra_finishes_fast": int(local_finishes[winner]),
        "elapsed_seconds_training_and_selection": perf_counter() - start_time,
        "champion": candidates[winner].tolist(),
    }


def assess(trial, courses):
    genome = np.asarray([trial["champion"]])
    outcomes = {}
    for name, course in courses.items():
        run = rollout(genome, course, symmetric=trial["architecture"] == "symmetric")
        outcomes[name] = {key: run[key] for key in ("reason", "ticks", "progress", "interventions")}
    return outcomes


def summarize(results, holdout_names):
    arms = {}
    for focus in (False, True):
        for architecture in ("symmetric", "free"):
            label = f"{architecture}_{'corners' if focus else 'restarts'}"
            rows = [row for row in results if row["architecture"] == architecture and row["corner_focus"] == focus]
            by_seed = {str(row["seed"]): sum(row["outcomes"][name]["reason"] == "finished"
                                             for name in holdout_names) for row in rows}
            arms[label] = {
                "seeds": len(rows),
                "training_finishes": sum(row["outcomes"]["training"]["reason"] == "finished" for row in rows),
                "holdout_finish_by_course": {name: sum(row["outcomes"][name]["reason"] == "finished"
                                                      for row in rows) for name in holdout_names},
                "holdout_finishes": sum(by_seed.values()),
                "holdout_body_contacts": sum(row["outcomes"][name]["reason"] == "body_contact"
                                             for row in rows for name in holdout_names),
                "all_holdouts_seeds": [int(seed) for seed, count in by_seed.items() if count == len(holdout_names)],
                "holdout_finishes_by_seed": by_seed,
                "budget_totals": {stage: {key: sum(row[stage][key] for row in rows)
                                          for key in rows[0][stage]}
                                  for stage in ("training", "selection", "nominal_verification")},
                "median_training_ticks": float(np.median([row["training"]["ticks"] for row in rows])),
            }
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds, generations, population = (SEEDS[:1], 2, 12) if args.smoke else (SEEDS, GENERATIONS, POPULATION)
    name = "single_course_corners_verified_smoke" if args.smoke else "single_course_corners_verified"
    directory = ROOT / "results" / name
    directory.mkdir(exist_ok=True)
    with np.load(MASKS) as archive:
        _, _, start, goal = SPECS["map6"]
        course = Corridor(archive["map6"], start, goal)
    corners = corner_trials(course)
    holdouts = {name: make_course(*settings, entry_extension=2) for name, settings in HOLDOUTS.items()}
    manifest = {
        "seeds": seeds, "generations": generations, "population": population, "pose_samples": POSE_SAMPLES,
        "selection_verification": "First ranked candidate independently replayed to a nominal training finish",
        "training_mask_sha256": sha256(course.mask.tobytes()).hexdigest(),
        "corner_route_indices": CORNERS,
        "corner_reference_witness_ticks": [corner_feasibility(trial, course.route) for trial in corners],
        "holdouts": HOLDOUTS,
        "holdout_masks_sha256": {name: sha256(layout.mask.tobytes()).hexdigest()
                                 for name, layout in holdouts.items()},
        "holdout_reference_witness_ticks": {name: feasibility(layout) for name, layout in holdouts.items()},
        "code_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in
                        ("single_course_corners.py", "corridor_search.py", "pilot.py", "footprint_guard.py",
                         "paper_corridors.py", "fresh_guard_check.py", "transfer_experiment.py",
                         "source_reproduction.py")},
        "numpy_version": np.__version__,
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Configuration changed. Move the previous results before starting a new run.")
    save_json(manifest_path, manifest)
    print("Recorded feasible training starts and fixed holdouts", manifest["holdout_reference_witness_ticks"], flush=True)
    jobs = [(architecture, seed, focus) for seed in seeds for focus in (False, True)
            for architecture in ("symmetric", "free")]
    paths = {job: directory / f"weights_{job[0]}_{'corners' if job[2] else 'restarts'}_{job[1]}.json"
             for job in jobs}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(train, course, *job, generations, population): job
                   for job, path in paths.items() if not path.exists()}
        for future in as_completed(pending):
            job = pending[future]
            result = future.result()
            save_json(paths[job], result)
            print("Trained", job, result["training"], flush=True)
    rows = [json.loads(paths[job].read_text()) for job in jobs]
    save_json(directory / "frozen_weights_sha256.json",
              {path.name: sha256(path.read_bytes()).hexdigest() for path in paths.values()})
    test_courses = {"training": course, **holdouts}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(assess, row, test_courses) for row in rows]
        for row, future in zip(rows, pending):
            row["outcomes"] = future.result()
            del row["champion"]
    save_json(directory / "outcomes.json", rows)
    summary = summarize(rows, holdouts)
    save_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
