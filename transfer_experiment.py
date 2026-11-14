"""Evolve on one corridor and measure geometric and left–right transfer separately."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
import json

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, distance_transform_edt, map_coordinates,
    maximum_filter, minimum_filter,
)

import pilot
from paper_corridors import Corridor, ROOT, SCALE, drive, load_corridors
from run_experiments import save_json
from source_reproduction import OPTIONS, Condition
from verify_source_controller import interpolated_contacts


CONDITIONS = {
    "symmetric": Condition(),
    "free": Condition("free", symmetric=False),
    "free_small": Condition("free_small", symmetric=False, hidden=8),
}
PROJECTED_CONTEXT = Condition("projected_context", symmetric=False, projected=True)
SEEDS = tuple(range(7310, 7322))
GENERATIONS = 25
POPULATION = 200


def restart_scenarios(course):
    scenarios = []
    for fraction in (0.2, 0.4, 0.6, 0.8):
        index = round(fraction * (len(course.route) - 1))
        tangent = course.route[index + 8] - course.route[index - 8]
        heading = np.arctan2(tangent[1], tangent[0])
        normal = pilot.direction(heading + np.pi / 2)
        for side in (-1, 1):
            scenario = copy(course)
            scenario.start = course.route[index] + side * 0.6 * normal
            scenario.heading = heading + side * np.deg2rad(10)
            x, y = np.rint(scenario.start / [SCALE, -SCALE]).astype(int)
            scenario.length = float(course.distance_to_goal[y, x])
            body = scenario.start + OPTIONS["vehicle"].body_offset * pilot.direction(scenario.heading)
            if not np.isfinite(scenario.length) or pilot.collides(body[None, :], np.array([scenario.heading]), course)[0]:
                raise ValueError(f"Invalid Figure 9 restart at route fraction {fraction}, side {side}")
            scenarios.append(scenario)
    return scenarios


def mirrored(course):
    last_row = course.mask.shape[0] - 1
    start_x, start_y = course.start_pixel
    goal_x, goal_y = course.goal_pixel
    result = Corridor(np.flipud(course.mask).copy(), (start_x, last_row - start_y),
                      (goal_x, last_row - goal_y))
    result.heading = -course.heading
    result.distance_to_goal = np.flipud(course.distance_to_goal).copy()
    result.length = course.length
    result.max_ticks = course.max_ticks
    return result


def guide_targets(course):
    """For each raster position, look two wheelbases ahead on Figure 9's route."""
    route = course.route
    path_length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    preview = np.minimum(np.searchsorted(path_length, path_length + 2 * pilot.WHEELBASE), len(route) - 1)
    x, y = np.rint(route / [SCALE, -SCALE]).astype(int).T
    nodes = np.full(course.mask.shape, -1, dtype=int)
    nodes[y, x] = np.arange(len(route))
    nearest = distance_transform_edt(nodes < 0, return_distances=False, return_indices=True)
    return route[preview[nodes[tuple(nearest)]]]


def layout_variants(course):
    """Vary one Figure 9 layout without importing any test-track geometry."""
    row, column = np.indices(course.mask.shape, dtype=float)
    distance = course.distance_to_goal
    free = course.mask & np.isfinite(distance)
    near = minimum_filter(np.where(free, distance, np.inf), size=5)
    far = maximum_filter(np.where(free, distance, -np.inf), size=5)
    # Keep the dividing walls between spatially close but distant track segments.
    wider = course.mask | (binary_dilation(course.mask) & (far - near < 8))
    masks = [binary_erosion(course.mask, border_value=1), wider]
    for sign in (-1, 1):
        offsets = [row + sign * 1.5 * np.sin(2 * np.pi * column / 96),
                   column + sign * 1.5 * np.sin(2 * np.pi * row / 96)]
        masks.append(map_coordinates(course.mask, offsets, order=0, mode="nearest"))
    variants = [Corridor(mask, course.start_pixel, course.goal_pixel) for mask in masks]
    for variant in variants:
        variant.heading = course.heading
        if not 0.95 * course.length <= variant.length <= 1.05 * course.length:
            raise ValueError("A training variant changed the Figure 9 route length beyond tolerance.")
    return variants


def training_scores(prepared, genomes, scenario, network, padding, targets, pose_samples, clearance_cost):
    result = prepared.evaluate(genomes, OPTIONS["vehicle"], padding, scenario=scenario, targets=targets,
                               pose_samples=pose_samples, clearance_cost=clearance_cost, **network)
    progress, finished, ticks = result[:3]
    scores = pilot.fitness(progress * scenario.length, ticks) / scenario.length
    if targets is not None:
        scores /= 1 + result[3] / np.deg2rad(30)
    if clearance_cost:
        scores /= 1 + 2 * result[4]
    return scores, finished


def train(course, seed, condition, restarts, generations, population, padding=0.0, guided=False,
          pose_samples=1, vary_layout=False, adaptive_speed=False, clearance_cost=False,
          verify_selection=False, audit_learning=False, *, mutation_probability=0.10, record_history=False,
          optimizer=None):
    from corridor_search import PreparedCourse

    rng = np.random.default_rng(seed)
    scenario_rng = np.random.default_rng([731, seed])
    ga = pilot.GASettings(population=population, mutation_probability_per_gene=mutation_probability)
    network = condition.network()
    if adaptive_speed:
        network["adaptive_speed"] = True
    genes = (14 if condition.symmetric else 27) * condition.hidden
    genomes = rng.uniform(-1, 1, (population, genes)) if optimizer is None else np.asarray(optimizer.ask())
    prepared = PreparedCourse(course)
    targets = guide_targets(course) if guided else None
    nominal = (prepared, course)
    pool = [(prepared, scenario) for scenario in restart_scenarios(course)] if restarts else []
    evaluation_pools = [pool]
    if vary_layout:
        evaluation_pools = []
        for variant in layout_variants(course):
            prepared_variant = PreparedCourse(variant)
            evaluation_pools.append([(prepared_variant, scenario) for scenario in restart_scenarios(variant)])
    shortlist = []
    first_finish = None
    audit = []
    witness = None
    history = []
    first_verified = None
    for generation in range(1, generations + 1):
        scenarios = [nominal]
        if restarts:
            for offset in (0, 4):
                chosen = evaluation_pools[scenario_rng.integers(len(evaluation_pools))] if vary_layout else pool
                scenarios.append(chosen[scenario_rng.integers(offset, offset + 4)])
        scores = np.zeros(population)
        for evaluator, scenario in scenarios:
            score, finished = training_scores(evaluator, genomes, scenario, network, padding, targets,
                                              pose_samples, clearance_cost)
            scores += score
            if scenario is course and audit_learning:
                entry, witness = audit_population(genomes, course, condition, generation, finished)
                audit.append(entry)
            if scenario is course and record_history and first_verified is None:
                _, verified_witness = audit_population(genomes, course, condition, generation, finished)
                if verified_witness is not None:
                    first_verified = generation
            if scenario is course and not audit_learning and first_finish is None and finished.any():
                _, verified, _, _ = drive(genomes[finished], course=course, **network, **OPTIONS)
                if verified.any():
                    first_finish = generation
        if record_history:
            history.append({"generation": generation, "best_fitness": float(scores.max()),
                            "median_fitness": float(np.median(scores))})
        if audit_learning and (witness is not None or generation == generations):
            return {
                "seed": seed, "architecture": condition.id, "population_size": population,
                "first_verified_population": generation if witness is not None else None,
                "witness": witness, "populations": audit,
            }
        top = np.argsort(-scores, kind="stable")[:3]
        shortlist.extend(genomes[top].copy())
        if optimizer is not None:
            optimizer.tell(genomes, -scores)
        if generation < generations:
            genomes = (pilot.offspring(genomes, scores, rng, ga) if optimizer is None
                       else np.asarray(optimizer.ask()))
    candidates = np.unique(np.asarray(shortlist), axis=0)
    scores = np.zeros(len(candidates))
    finish_counts = np.zeros(len(candidates), dtype=int)
    selection = [nominal, *(evaluation_pools[i % len(evaluation_pools)][i] for i in range(len(pool)))]
    for evaluator, scenario in selection:
        score, finished = training_scores(evaluator, candidates, scenario, network, padding, targets,
                                          pose_samples, clearance_cost)
        scores += score
        finish_counts += finished
        if scenario is course:
            nominal_finishes = finished
    rank = np.lexsort((-scores, -finish_counts, ~nominal_finishes))
    winner = rank[0]
    verified = None
    if verify_selection:
        winner, verified = select_verified(candidates, rank, [scenario for _, scenario in selection], condition)
    result = {
        "seed": seed, "architecture": condition.id, "restarts": restarts,
        "generations": generations, "population": population, "gene_count": genes,
        "training_padding_m": padding, "training_guide": guided,
        "training_pose_samples": pose_samples, "training_layout_variants": vary_layout,
        "adaptive_speed": adaptive_speed,
        "training_clearance_cost": clearance_cost,
        "first_nominal_finish_population": first_finish,
        "selection_finishes": int(finish_counts[winner]), "selection_episodes": 1 + len(pool),
        "champion": candidates[winner].tolist(),
    }
    if verified is not None:
        result["verified_training_finishes"] = verified
        result["selection_candidate_changed"] = bool(winner != rank[0])
    if record_history:
        result.update(first_verified_population=first_verified, fitness_history=history)
        if optimizer is None:
            result["mutation_probability_per_gene"] = mutation_probability
    return result


def assess(genome, course, condition, *, adaptive_speed=False):
    network = condition.network()
    if adaptive_speed:
        network["adaptive_speed"] = True
    _, finished, progress, path = drive(genome, course=course, trace=True, **network, **OPTIONS)
    contacts = None
    if finished[0]:
        contacts = interpolated_contacts(genome, course, np.asarray(path), **network)
    return {"finished": bool(finished[0] and contacts == 0), "raw_finish": bool(finished[0]),
            "progress": float(progress[0]), "ticks": len(path) - 1, "interpolated_contacts": contacts}


def audit_population(genomes, course, condition, generation, fast_finished):
    _, reference_finished, _, _ = drive(genomes, course=course, **condition.network(), **OPTIONS)
    counts = {
        "population": generation, "fast_finishes": int(fast_finished.sum()),
        "reference_step_end_finishes": int(reference_finished.sum()),
        "reference_finishes_missed_by_fast": int(np.count_nonzero(reference_finished & ~fast_finished)),
        "contact_rejections_before_first_pass": 0,
    }
    for index in np.flatnonzero(reference_finished):
        candidate = genomes[index:index + 1]
        replay = assess(candidate, course, condition)
        if replay["finished"]:
            return counts, {"index": int(index), "ticks": replay["ticks"], "weights": candidate[0].tolist()}
        counts["contact_rejections_before_first_pass"] += 1
    return counts, None


def select_verified(candidates, rank, scenarios, condition):
    """Prefer nominal finish, then verified restarts, then the original rank."""
    best_index = rank[0]
    best_outcomes = [False] * len(scenarios)
    best_key = (False, -1)
    for index in rank:
        genome = candidates[index:index + 1]
        nominal = assess(genome, scenarios[0], condition)["finished"]
        if not nominal and best_key[0]:
            continue
        outcomes = [nominal, *(assess(genome, scenario, condition)["finished"] for scenario in scenarios[1:])]
        key = (nominal, sum(outcomes))
        if key > best_key:
            best_index, best_outcomes, best_key = index, outcomes, key
        if all(outcomes):
            break
    return best_index, best_outcomes


def evaluate(directory, runs):
    courses = load_corridors()
    reflections = {name: mirrored(course) for name, course in courses.items()}
    results = []
    for label in runs:
        trial = json.loads((directory / f"weights_{label}.json").read_text())
        genome = np.asarray([trial.pop("champion")])
        architecture = trial["architecture"]
        condition = PROJECTED_CONTEXT if architecture == PROJECTED_CONTEXT.id else CONDITIONS[architecture]
        adaptive_speed = trial.get("adaptive_speed", False)
        trial["courses"] = {name: {"original": assess(genome, course, condition, adaptive_speed=adaptive_speed),
                                   "mirror": assess(genome, reflections[name], condition, adaptive_speed=adaptive_speed)}
                            for name, course in courses.items()}
        save_json(directory / f"outcome_{label}.json", trial)
        results.append(trial)
        print(f"Checked {label}", flush=True)
    return results


def summarize(results):
    summary = {}
    for architecture, restarts in dict.fromkeys((row["architecture"], row["restarts"]) for row in results):
        rows = [row for row in results if (row["architecture"], row["restarts"]) == (architecture, restarts)]
        label = f"{architecture}_{'restarts' if restarts else 'fixed'}"
        summary[label] = {
            "seeds": len(rows),
            "nominal_training_finishes": sum(row["courses"]["map6"]["original"]["finished"] for row in rows),
            "mirrored_training_finishes": sum(row["courses"]["map6"]["mirror"]["finished"] for row in rows),
            "heldout_original_finishes": sum(row["courses"][name]["original"]["finished"]
                                            for row in rows for name in ("map7", "map8", "map9")),
            "heldout_mirror_finishes": sum(row["courses"][name]["mirror"]["finished"]
                                          for row in rows for name in ("map7", "map8", "map9")),
            "all_three_originals": [row["seed"] for row in rows
                                    if all(row["courses"][name]["original"]["finished"]
                                           for name in ("map7", "map8", "map9"))],
            "all_six_heldout": [row["seed"] for row in rows
                                if all(row["courses"][name][side]["finished"]
                                       for name in ("map7", "map8", "map9") for side in ("original", "mirror"))],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Replay all archived weights")
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--clearance", action="store_true", help="Train with a 0.25 m body margin")
    experiment.add_argument("--guided", action="store_true", help="Train with a reference trajectory score")
    experiment.add_argument("--swept", action="store_true", help="Check intermediate poses during training")
    experiment.add_argument("--layout", action="store_true", help="Train on Figure 9 variants and check intermediate poses")
    experiment.add_argument("--context", action="store_true", help="Train output-projected symmetry with all 25 inputs")
    experiment.add_argument("--speed", action="store_true", help="Train and test with a LiDAR-based speed rule")
    experiment.add_argument("--near-miss", action="store_true", help="Penalize near-wall body poses during training")
    experiment.add_argument("--verified-selection", action="store_true",
                            help="Verify all nine training replays before selecting a network")
    parser.add_argument("--free", action="store_true", help="Use the unconstrained architecture for verified selection")
    args = parser.parse_args()
    if args.free and not args.verified_selection:
        parser.error("--free requires --verified-selection")
    name = "transfer"
    for option in ("guided", "clearance", "swept", "layout", "context", "speed", "near_miss",
                   "verified_selection"):
        if getattr(args, option):
            name = f"transfer_{option}"
    if args.free:
        name += "_free"
    directory = ROOT / "results" / (f"{name}_smoke" if args.smoke else name)
    directory.mkdir(parents=True, exist_ok=True)
    seeds, generations, population = ((SEEDS[0],), 2, 12) if args.smoke else (SEEDS, GENERATIONS, POPULATION)
    arms = [(condition, True) for condition in CONDITIONS.values()] + [(CONDITIONS["symmetric"], False)]
    if name != "transfer":
        condition = PROJECTED_CONTEXT if args.context else CONDITIONS["free" if args.free else "symmetric"]
        arms = [(condition, True)]
    jobs = {f"{condition.id}_{'restarts' if restarts else 'fixed'}_{seed}": (seed, condition, restarts)
            for condition, restarts in arms for seed in seeds}
    if not args.evaluate:
        course = load_corridors()["map6"]
        for layout in [course, *(layout_variants(course) if args.layout else [])]:
            restart_scenarios(layout)
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = {
                pool.submit(train, course, *job, generations, population,
                            padding=0.25 if args.clearance else 0.0, guided=args.guided,
                            pose_samples=4 if name in ("transfer_swept", "transfer_layout", "transfer_context",
                                                       "transfer_speed", "transfer_near_miss") else 1,
                            vary_layout=args.layout, adaptive_speed=args.speed, clearance_cost=args.near_miss,
                            verify_selection=args.verified_selection): label
                for label, job in jobs.items() if not (directory / f"weights_{label}.json").exists()
            }
            for future in as_completed(pending):
                label = pending[future]
                save_json(directory / f"weights_{label}.json", future.result())
                print(f"Frozen {label}", flush=True)
    if not args.smoke or args.evaluate:
        summary = summarize(evaluate(directory, jobs))
        save_json(directory / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
