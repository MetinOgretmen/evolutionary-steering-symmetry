"""Run the frozen AE4350 experiment and save one JSON result per GA run."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

import pilot


ROOT = Path(__file__).resolve().parent


def make_scenarios(specs, geometry, mirror=False):
    scenarios = []
    for spec in specs:
        obstacles = np.asarray(spec["obstacles_m"], dtype=float)
        variants = [(spec["id"], obstacles)]
        if mirror:
            reflected = obstacles.copy()
            reflected[:, [1, 3]] = geometry["width_m"] - obstacles[:, [3, 1]]
            variants.append((f'{spec["id"]}_mirror', reflected))
        for label, boxes in variants:
            course = pilot.Course(geometry["length_m"], geometry["finish_x_m"], boxes)
            for index, y in enumerate(geometry["start_y_m"]):
                start = np.array([geometry["start_x_m"], y])
                scenarios.append((f"{label}_start{index + 1}", course, start))
    return scenarios


def validate_config(config):
    geometry, vehicle, sensor = (config[key] for key in ("geometry", "vehicle", "sensor"))
    # The vehicle and angular grid are shared by all runs in this study.
    fixed = (
        (geometry["width_m"], pilot.COURSE_WIDTH),
        (geometry["start_heading_deg"], 0.0),
        (vehicle["wheelbase_m"], pilot.WHEELBASE),
        (vehicle["length_m"], 2 * pilot.HALF_LENGTH),
        (vehicle["width_m"], 2 * pilot.HALF_WIDTH),
        (vehicle["speed_mps"], 4.0),
        (vehicle["dt_s"], 0.1),
        (vehicle["speed_mps"] * vehicle["dt_s"], pilot.STEP_DISTANCE),
        (vehicle["steering_limit_deg"], np.rad2deg(pilot.STEERING_LIMIT)),
        (vehicle["max_ticks"], pilot.MAX_TICKS),
        (sensor["range_m"], pilot.SENSOR_RANGE),
        (sensor["field_of_view_deg"], np.rad2deg(np.ptp(pilot.BEAM_ANGLES))),
        (sensor["input_count"], len(pilot.BEAM_ANGLES)),
        (config["ga"]["crossover_probability"], 1.0),
    )
    if any(not np.isclose(value, expected) for value, expected in fixed):
        raise ValueError("Config does not match the fixed vehicle, sensor, or crossover settings.")
    for condition in config["conditions"]:
        elites = condition["elites"]
        population = config["ga"]["population"]
        if (condition["physical_beams"] not in (7, 25) or condition["hidden"] < 1
                or not 0 <= elites < population or (population - elites) % 2):
            raise ValueError(f"Unsupported condition {condition['id']}")


def test_controller(champion, network, scenarios, sensor, seed):
    episodes = []
    for index, (scenario_id, course, start) in enumerate(scenarios):
        for sigma in sensor["test_noise_std"]:
            repeats = sensor["noisy_repeats"] if sigma else 1
            for repeat in range(repeats):
                noise = None
                if sigma:
                    rng = np.random.default_rng([sensor["noise_seed"], seed, index, repeat])
                    noise = sigma * rng.standard_normal((pilot.MAX_TICKS, len(pilot.BEAM_ANGLES)))
                score, finished, _ = pilot.evaluate(champion, **network, course=course,
                                                     start=start, noise=noise)
                episodes.append({
                    "scenario": scenario_id,
                    "noise_std": sigma,
                    "repeat": repeat,
                    "finished": bool(finished[0]),
                    "fitness": float(score[0]),
                })
    return episodes


def run_one(condition, seed, generations, ga, training, test, sensor):
    network = {key: condition[key] for key in ("symmetric", "hidden", "physical_beams")}
    run = pilot.search(generations=generations, seed=seed, scenarios=training,
                       ga=ga, elites=condition["elites"], **network)
    return {
        "condition": condition["id"],
        "seed": seed,
        "fitness_history": run.history.tolist(),
        "initial_full_successes": run.initial_full_successes,
        "first_full_success_population": run.first_full_success,
        "champion": run.champion[0].tolist(),
        "champion_training_finishes": run.champion_finishes.tolist(),
        "test_episodes": test_controller(run.champion, network, test, sensor, seed),
    }


def save_json(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            json.dump(data, temp, allow_nan=False, separators=(",", ":"))
            temp.write("\n")
            temp.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def check_experiment(config, training, test, ga):
    pilot.check_pilot()
    assert len(training) == 4 and len(test) == 12
    _, course, start = test[0]
    _, mirrored, reflected_start = test[3]
    heading = np.array([0.0])
    full = pilot.sense(start[None, :], heading, course)[0]
    low = pilot.sense(start[None, :], heading, course, physical_beams=7)[0]
    samples = pilot.LOW_RESOLUTION_INDICES
    np.testing.assert_allclose(low, np.interp(np.arange(25), samples, full[samples]))
    for beams in (7, 25):
        left = pilot.sense(start[None, :], heading, course, physical_beams=beams)
        right = pilot.sense(reflected_start[None, :], heading, mirrored, physical_beams=beams)
        np.testing.assert_allclose(left[:, ::-1], right, atol=1e-12)
    noise = np.linspace(-2, 2, len(full))
    low_noisy = pilot.sense(start[None, :], heading, course, physical_beams=7, relative_noise=noise)[0]
    expected = np.clip(full[samples] * (1 + noise[samples]), 0, 1)
    np.testing.assert_allclose(low_noisy, np.interp(np.arange(25), samples, expected))
    rng = np.random.default_rng(4350)
    genomes = rng.uniform(-1, 1, (ga.population, 210))
    scores = rng.random(ga.population)
    next_population = pilot.offspring(genomes, scores, rng, ga, elites=10)
    assert next_population.shape == genomes.shape
    np.testing.assert_array_equal(next_population[:10], genomes[np.argsort(-scores)[:10]])
    for condition in config["conditions"]:
        # Smoke test on a training course so trained controllers stay off the test set.
        stand_in = [("smoke", training[0][0], training[0][1])]
        result = run_one(condition, 4350, 2, ga, training, stand_in, config["sensor"])
        assert len(result["fitness_history"]) == 2
        assert len(result["champion_training_finishes"]) == len(training)
        assert len(result["test_episodes"]) == 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            save_json(path, result)
            assert json.loads(path.read_text()) == result
    print("All seven conditions passed the short check (sensor, mirror, elitism, and JSON output).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--check", action="store_true", help="Check the pipeline using training courses")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config = json.loads((ROOT / "experiment_config.json").read_text())
    validate_config(config)
    ga_config = config["ga"]
    ga = pilot.GASettings(**{key: ga_config[key] for key in pilot.GASettings.__dataclass_fields__})
    training = make_scenarios(config["training_courses"], config["geometry"])
    training_pairs = [(course, start) for _, course, start in training]
    test = make_scenarios(config["test_courses"], config["geometry"], config["mirror_test_courses"])
    if args.check:
        check_experiment(config, training_pairs, test, ga)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("pilot.py", "run_experiments.py")
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("This output folder contains results from a different configuration or code version.")
    else:
        save_json(manifest_path, manifest)

    jobs = [(condition, seed, args.output / f"{condition['id']}_{seed}.json")
            for condition in config["conditions"] for seed in ga_config["seeds"]]
    pending = [(condition, seed, path) for condition, seed, path in jobs if not path.exists()]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} runs already complete", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, condition, seed, ga_config["populations_evaluated"], ga,
                        training_pairs, test, config["sensor"]): path
            for condition, seed, path in pending
        }
        for count, future in enumerate(as_completed(futures), len(jobs) - len(pending) + 1):
            path = futures[future]
            result = future.result()
            save_json(path, result)
            print(f"{count}/{len(jobs)} {path.stem}", flush=True)


if __name__ == "__main__":
    main()
