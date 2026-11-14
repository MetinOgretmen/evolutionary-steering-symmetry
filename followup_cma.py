"""Train bounded full-covariance CMA-ES under the main GA evaluation budget."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json

import cma
import numpy as np

from followup_courses import DESTINATION as GEOMETRY, DEPTHS, WIDTHS, layouts, replay
from followup_initial import write_csv
from followup_mutation import TESTS, seed_interval
from fresh_guard_check import ARCHIVES
from paper_corridors import ROOT, load_corridors
from run_experiments import save_json
from transfer_experiment import CONDITIONS, GENERATIONS, POPULATION, SEEDS, assess, train


DESTINATION = ROOT / "results" / "followup_cma"
SIGMA = 1 / np.sqrt(3)
ARCHITECTURES = ("symmetric", "free")
ARMS = tuple(f"{method}_{arch}" for method in ("ga", "cma") for arch in ARCHITECTURES)


def run_seed(architecture, seed, courses, generations, population):
    condition = CONDITIONS[architecture]
    genes = (14 if condition.symmetric else 27) * condition.hidden
    optimizer = cma.CMAEvolutionStrategy(np.zeros(genes), SIGMA, {
        "seed": seed, "popsize": population, "bounds": [-1.0, 1.0],
        "CMA_diagonal": 0, "maxiter": generations, "verbose": -9, "verb_log": 0,
    })
    result = train(courses["map6"], seed, condition, True, generations, population,
                   verify_selection=True, record_history=True, optimizer=optimizer)
    genome = np.asarray([result["champion"]])
    assert optimizer.countevals == generations * population
    assert optimizer.countiter == generations
    assert np.all(np.abs(genome) <= 1)
    result.update(optimizer="CMA-ES", pycma_version=cma.__version__, initial_sigma=float(SIGMA),
                  final_sigma=float(optimizer.sigma), candidate_evaluations=optimizer.countevals,
                  training_drives=3 * optimizer.countevals,
                  courses={name: assess(genome, courses[name], condition) for name in TESTS})
    return result


def frozen_replay(directory, rows, workers):
    with (GEOMETRY / "course_manifest.csv").open() as stream:
        eligible = {r["course"] for r in csv.DictReader(stream) if r["eligible"] == "True"}
    specs = [spec for spec in layouts() if spec["course"] in eligible]
    assert {spec["course"] for spec in specs} == eligible and len(specs) == 148
    drivers = [(row["architecture"], row["seed"], np.asarray([row["champion"]])) for row in rows]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(replay, spec, drivers) for spec in specs]
        for index, job in enumerate(jobs, 1):
            output.extend(job.result())
            if index % 24 == 0 or index == len(specs):
                print(f"CMA-ES replay {index}/{len(specs)}", flush=True)
    write_csv(directory / "procedural_drives.csv", output)
    return output


def ga_outcome(architecture, seed):
    label = f"{architecture}_restarts_{seed}"
    archive = ROOT / "results" / ARCHIVES[architecture]
    row = json.loads((archive / f"outcome_{label}.json").read_text())
    audit = json.loads((ROOT / "results" / "learning_speed_audit" / f"audit_{architecture}_{seed}.json").read_text())
    row["first_verified_population"] = audit["first_verified_population"]
    row["courses"] = {name: row["courses"][name]["original"] for name in TESTS}
    return row


def summarize(directory, rows, procedural):
    with (GEOMETRY / "drives.csv").open() as stream:
        original = [r for r in csv.DictReader(stream) if r["arm"] in ARCHITECTURES]
    for method, drives in (("ga", original), ("cma", procedural)):
        for row in drives:
            row["arm"] = f"{method}_{row['arm']}"
            row["finish"] = str(row["finish"]) == "True"
            row["seed"] = int(row["seed"])
            row["width_m"] = float(row["width_m"])
            row["depth_m"] = float(row["depth_m"])
    all_drives = original + procedural
    champions = {("cma", row["architecture"], row["seed"]): row for row in rows}
    for row in rows:
        assert row["candidate_evaluations"] == GENERATIONS * POPULATION
        assert row["training_drives"] == 3 * GENERATIONS * POPULATION
        assert len(row["fitness_history"]) == GENERATIONS
    write_csv(directory / "fitness_history.csv",
              [{"architecture": row["architecture"], "seed": row["seed"], **generation}
               for row in rows for generation in row["fitness_history"]])
    champions.update({("ga", arch, seed): ga_outcome(arch, seed)
                      for arch in ARCHITECTURES for seed in SEEDS})
    by_run = []
    for (method, arch, seed), row in sorted(champions.items()):
        arm = f"{method}_{arch}"
        drives = [r for r in all_drives if r["arm"] == arm and r["seed"] == seed]
        assert len(drives) == 148 and len({r["course"] for r in drives}) == 148
        abc = [row["courses"][name]["finished"] for name in TESTS]
        by_run.append({"arm": arm, "seed": seed, "first_verified_generation": row["first_verified_population"],
                       "nominal_T_finished": row["verified_training_finishes"][0],
                       "T_starts_finished": sum(row["verified_training_finishes"]),
                       **dict(zip(("A", "B", "C"), abc, strict=True)),
                       "ABC_finishes": sum(abc), "all_ABC": all(abc),
                       "procedural_finishes": sum(r["finish"] for r in drives)})
    write_csv(directory / "comparison_by_run.csv", by_run)
    records = {(r["arm"], r["seed"]): r for r in by_run}

    def gap(metric, method):
        return np.asarray([int(records[f"{method}_symmetric", seed][metric])
                           - int(records[f"{method}_free", seed][metric]) for seed in SEEDS])

    def comparisons(metric, scale):
        ga, cma_gap = (gap(metric, method) / scale for method in ("ga", "cma"))
        return {"GA_symmetric_minus_free": seed_interval(ga),
                "CMA_symmetric_minus_free": seed_interval(cma_gap),
                "CMA_gap_minus_GA_gap": seed_interval(cma_gap - ga)}

    summary = {"ABC_courses_per_driver": comparisons("ABC_finishes", 1),
               "procedural_finish_fraction": comparisons("procedural_finishes", 148),
               "arms": {}}
    for arm in ARMS:
        subset = [r for r in by_run if r["arm"] == arm]
        found = [r["first_verified_generation"] for r in subset if r["first_verified_generation"] is not None]
        summary["arms"][arm] = {
            "T_finished": sum(r["nominal_T_finished"] for r in subset),
            "T_starts_finished": sum(r["T_starts_finished"] for r in subset),
            "found_driver_during_training": len(found),
            "median_first_verified_generation_among_successes": float(np.median(found)) if found else None,
            "initial_pools_with_finish": found.count(1),
            "ABC_by_course": [sum(r[name] for r in subset) for name in ("A", "B", "C")],
            "ABC_total": sum(r["ABC_finishes"] for r in subset),
            "ABC_all_three": sum(r["all_ABC"] for r in subset),
            "procedural_total": sum(r["procedural_finishes"] for r in subset),
        }
    geometry = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for arm in ARMS:
                cell = [r for r in all_drives if (r["width_m"], r["depth_m"], r["arm"]) == (width, depth, arm)]
                if cell:
                    geometry.append({"width_m": width, "depth_m": depth, "arm": arm, "drives": len(cell),
                                     "finishes": sum(r["finish"] for r in cell)})
    write_csv(directory / "comparison_by_geometry.csv", geometry)
    save_json(directory / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    directory = DESTINATION.with_name(DESTINATION.name + "_smoke") if args.smoke else DESTINATION
    directory.mkdir(parents=True, exist_ok=True)
    courses = load_corridors()
    seeds, generations, population = ((SEEDS[0],), 2, 12) if args.smoke else (SEEDS, GENERATIONS, POPULATION)
    paths = {(arch, seed): directory / f"run_{arch}_{seed}.json" for arch in ARCHITECTURES for seed in seeds}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_seed, *key, courses, generations, population): key
                   for key, path in paths.items() if not path.exists()}
        for task in as_completed(pending):
            key = pending[task]
            row = task.result()
            save_json(paths[key], row)
            print(*key, row["first_verified_population"], sum(r["finished"] for r in row["courses"].values()), flush=True)
    if not args.smoke:
        rows = [json.loads(path.read_text()) for path in paths.values()]
        procedural_path = directory / "procedural_drives.csv"
        if procedural_path.exists():
            with procedural_path.open() as stream:
                procedural = list(csv.DictReader(stream))
        else:
            procedural = frozen_replay(directory, rows, args.workers)
        summarize(directory, rows, procedural)


if __name__ == "__main__":
    main()
