"""Re-evolve on Figure 9 with the sensor, body reference, and step from GA-NN-Car."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import pilot
from paper_corridors import MATLAB_VEHICLE, ROOT, SCALE, drive, load_corridors
from run_experiments import save_json


OPTIONS = {
    "steering_scale": 90 / 35,
    "input_scale": pilot.SENSOR_RANGE / SCALE,
    "vehicle": MATLAB_VEHICLE,
}


@dataclass(frozen=True)
class Condition:
    id: str = "symmetric"
    symmetric: bool = True
    hidden: int = 15
    physical_beams: int = 25
    elites: int = 0
    projected: bool = False

    def network(self):
        network = {"symmetric": self.symmetric, "hidden": self.hidden, "physical_beams": self.physical_beams}
        if self.projected:
            network["projected"] = True
        return network


def train_one(course, seed, generations, population, condition=Condition(), *, training_noise=0.0):
    from corridor_search import PreparedCourse

    rng = np.random.default_rng(seed)
    ga = pilot.GASettings(population=population)
    prepared = PreparedCourse(course)
    gene_count = (14 if condition.symmetric else 27) * condition.hidden
    genomes = rng.uniform(-1, 1, (population, gene_count))
    network = condition.network()
    best_fast = -np.inf
    shortlist = []
    history = []
    first_finish = None
    noise_rng = np.random.default_rng([8400, seed])
    noise_shape = (course.max_ticks, len(pilot.BEAM_ANGLES))
    for number in range(1, generations + 1):
        noise = training_noise * noise_rng.standard_normal(noise_shape) if training_noise else None
        progress, finished, ticks = prepared.evaluate(genomes, MATLAB_VEHICLE, noise=noise, **network)
        scores = pilot.fitness(progress * course.length, ticks)
        order = np.argsort(-scores, kind="stable")
        if scores[order[0]] > best_fast:
            best_fast = scores[order[0]]
            shortlist.extend(genomes[order[:3]].copy())
        if first_finish is None and finished.any():
            flagged = genomes[finished]
            _, verified, _, _ = drive(flagged, course=course, noise=noise, **network, **OPTIONS)
            if verified.any():
                first_finish = number
        history.append(float(best_fast))
        if number < generations:
            genomes = pilot.offspring(genomes, scores, rng, ga, condition.elites)
    candidates = np.asarray(shortlist)
    validation_noise = [None]
    if training_noise:
        validation_rng = np.random.default_rng([8401, seed])
        validation_noise = [training_noise * validation_rng.standard_normal(noise_shape) for _ in range(3)]
    validation = [drive(candidates, course=course, noise=noise, **network, **OPTIONS)[:3]
                  for noise in validation_noise]
    scores, finishes, progress = (np.asarray(metric) for metric in zip(*validation))
    scores = scores.mean(axis=0)
    winner = int(np.argmax(scores))
    trial = {
        "seed": seed,
        "condition": condition.id,
        "populations": generations,
        "population": population,
        "fitness_history": history,
        "first_verified_finish_population": first_finish,
        "champion": candidates[winner].tolist(),
        "training": {"finished": bool(finishes[:, winner].all()), "progress": float(progress[:, winner].mean())},
    }
    if training_noise:
        trial["training_noise"] = training_noise
        trial["training"]["selection_finishes"] = finishes[:, winner].tolist()
    return trial


def test_frozen(trial, tests, condition=Condition()):
    genome = np.asarray([trial["champion"]])
    return {
        name: {"finished": bool(finished[0]), "progress": float(progress[0]), "ticks": len(path) - 1}
        for name, course in tests.items()
        for _, finished, progress, path in [drive(genome, course=course, trace=True,
                                                **condition.network(), **OPTIONS)]
    }


def test_noise(trial, tests, condition, sigmas=(0.05, 0.10)):
    genome = np.asarray([trial["champion"]])
    outcomes = []
    for index, (name, course) in enumerate(tests.items()):
        for repeat in range(3):
            rng = np.random.default_rng([7400, trial["seed"], index, repeat])
            standard = rng.standard_normal((course.max_ticks, len(pilot.BEAM_ANGLES)))
            for sigma in sigmas:
                _, finished, progress, _ = drive(genome, course=course, noise=sigma * standard,
                                                **condition.network(), **OPTIONS)
                outcomes.append({"course": name, "sigma": sigma, "repeat": repeat,
                                 "finished": bool(finished[0]), "progress": float(progress[0])})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, default=25)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[6200, 6201, 6202])
    parser.add_argument("--matrix", action="store_true", help="Compare all seven conditions on the same vehicle")
    parser.add_argument("--check", action="store_true", help="Check accelerated steering against the Python simulator")
    parser.add_argument("--training-noise", type=float, default=0.0,
                        help="Train the baseline symmetric network on noisy Figure 9 ranges (relative SD)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.populations < 1 or args.population < 2 or args.population % 2:
        parser.error("Use at least one population and a positive even candidate count.")
    if not np.isfinite(args.training_noise) or args.training_noise < 0:
        parser.error("Training noise must be finite and nonnegative.")
    if args.training_noise and (args.matrix or args.check or args.output is None):
        parser.error("For noisy training, run the baseline condition with a separate --output directory.")
    courses = load_corridors()
    training = courses.pop("map6")
    config = json.loads((ROOT / "experiment_config.json").read_text())
    conditions = [Condition(**item) for item in config["conditions"]] if args.matrix or args.check else [Condition()]
    if args.check:
        check_steering(training, conditions)
        return
    directory = args.output or ROOT / "results" / ("source_matrix" if args.matrix else "source_alignment")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(condition, seed) for condition in conditions for seed in args.seeds]
    frozen = []
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        pending = {pool.submit(train_one, training, seed, args.populations, args.population, condition,
                               training_noise=args.training_noise):
                   condition for condition, seed in jobs}
        for index, future in enumerate(as_completed(pending), 1):
            condition = pending[future]
            trial = future.result()
            label = f"{condition.id}_{trial['seed']}" if args.matrix else str(trial["seed"])
            save_json(directory / f"weights_{label}.json", trial)
            frozen.append((condition, label, trial))
            print(f"{index}/{len(jobs)} frozen: {label}, first verified finish "
                  f"{trial['first_verified_finish_population']}, selected training {trial['training']}", flush=True)
    for condition, label, trial in frozen:
        trial["test"] = test_frozen(trial, courses, condition)
        if args.matrix or args.training_noise:
            sigmas = (0.05, 0.10, 0.20) if args.training_noise else (0.05, 0.10)
            trial["noise_test"] = test_noise(trial, courses, condition, sigmas)
        save_json(directory / f"outcome_{label}.json", trial)
        print(label, [name for name, result in trial["test"].items() if result["finished"]], flush=True)


def check_steering(course, conditions):
    from corridor_search import steering

    rng = np.random.default_rng(6201)
    checkpoints = course.route[::max(1, len(course.route) // 20)][:20]
    for condition in conditions:
        genome = rng.uniform(-1, 1, (1, (14 if condition.symmetric else 27) * condition.hidden))
        inputs, outputs = pilot.decode(genome, condition.symmetric, condition.hidden)
        for position in checkpoints:
            angle = rng.uniform(-np.pi, np.pi)
            heading = np.array([angle])
            sensor_position = position[None, :] - pilot.HALF_LENGTH * pilot.direction(heading)
            for noise in (None, rng.normal(0, 0.05, len(pilot.BEAM_ANGLES))):
                scan = pilot.sense(sensor_position, heading, course, physical_beams=condition.physical_beams,
                                   relative_noise=noise)
                expected = (90 / 35) * pilot.command(scan * OPTIONS["input_scale"], inputs, outputs,
                                                     condition.symmetric)[0]
                actual = steering(genome[0], course.vertical, course.horizontal, *position, angle,
                                  np.cos(angle), np.sin(angle), 0.0, relative_noise=noise,
                                  **condition.network())
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    print("All seven conditions match Python steering with clean and noisy readings at twenty positions.")


if __name__ == "__main__":
    main()
