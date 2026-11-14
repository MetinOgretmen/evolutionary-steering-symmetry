"""Train the unrestricted mutation-rate control and replay the archived comparison arms."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json

import numpy as np

from fresh_guard_check import ARCHIVES
from paper_corridors import ROOT, load_corridors
from run_experiments import save_json
from transfer_experiment import CONDITIONS, GENERATIONS, POPULATION, SEEDS, assess, train

DESTINATION = ROOT / "results" / "followup_mutation"
RATE = 0.052
TESTS = ("map7", "map8", "map9")


def run_seed(seed, courses, generations, population):
    condition = CONDITIONS["free"]
    result = train(courses["map6"], seed, condition, True, generations, population,
                   verify_selection=True, mutation_probability=RATE, record_history=True)
    genome = np.asarray([result["champion"]])
    result["courses"] = {name: assess(genome, courses[name], condition) for name in TESTS}
    return result


def seed_interval(differences):
    differences = np.asarray(differences)
    draws = np.random.default_rng(4350).choice(differences, size=(20000, len(differences)))
    return {"mean": float(differences.mean()),
            "paired_bootstrap_95": np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()}


def summarize(rows):
    output = []
    for row in rows:
        seed = row["seed"]
        for architecture, archive in ARCHIVES.items():
            original = json.loads((ROOT / "results" / archive / f"outcome_{architecture}_restarts_{seed}.json").read_text())
            audit = json.loads((ROOT / "results" / "learning_speed_audit" / f"audit_{architecture}_{seed}.json").read_text())
            output.append({"arm": architecture, "seed": seed,
                           "first_verified_population": audit["first_verified_population"],
                           "courses_finished": sum(original["courses"][name]["original"]["finished"] for name in TESTS)})
        output.append({"arm": "free_mutation_052", "seed": seed,
                       "first_verified_population": row["first_verified_population"],
                       "courses_finished": sum(row["courses"][name]["finished"] for name in TESTS)})
    arms = {arm: [r for r in output if r["arm"] == arm]
            for arm in ("symmetric", "free", "free_mutation_052")}
    differences = {}
    for control in ("free", "symmetric"):
        comparisons = list(zip(arms["free_mutation_052"], arms[control], strict=True))
        differences[control] = {
            "new_minus_control_courses_finished": seed_interval(
                [a["courses_finished"] - b["courses_finished"] for a, b in comparisons]),
            "new_minus_control_generation_capped_at_26": seed_interval(
                [(a["first_verified_population"] or 26) - (b["first_verified_population"] or 26)
                 for a, b in comparisons]),
        }
    return {"comparisons": differences, "runs": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    directory = DESTINATION.with_name(DESTINATION.name + "_smoke") if args.smoke else DESTINATION
    directory.mkdir(parents=True, exist_ok=True)
    courses = load_corridors()
    seeds, generations, population = ((SEEDS[0],), 2, 12) if args.smoke else (SEEDS, GENERATIONS, POPULATION)
    paths = {seed: directory / f"run_{seed}.json" for seed in seeds}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_seed, seed, courses, generations, population): seed
                   for seed in seeds if not paths[seed].exists()}
        for task in as_completed(pending):
            seed = pending[task]
            result = task.result()
            save_json(paths[seed], result)
            print(seed, result["first_verified_population"],
                  sum(row["finished"] for row in result["courses"].values()), flush=True)
    if not args.smoke:
        rows = [json.loads(paths[seed].read_text()) for seed in seeds]
        summary = summarize(rows)
        save_json(directory / "summary.json", summary)
        with (directory / "fitness_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("seed", "generation", "best_fitness", "median_fitness"))
            writer.writeheader()
            for row in rows:
                writer.writerows({"seed": row["seed"], **generation} for generation in row["fitness_history"])


if __name__ == "__main__":
    main()
