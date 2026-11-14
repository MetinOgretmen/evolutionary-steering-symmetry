"""Re-score the original candidate populations using ten body poses per step."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from statistics import median

from paper_corridors import ROOT, load_corridors
from run_experiments import save_json
from transfer_experiment import CONDITIONS, GENERATIONS, POPULATION, SEEDS, train


def main():
    directory = ROOT / "results" / "learning_speed_audit"
    directory.mkdir(exist_ok=True)
    course = load_corridors()["map6"]
    conditions = [CONDITIONS[name] for name in ("symmetric", "free")]
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(train, course, seed, condition, True, GENERATIONS, POPULATION,
                               audit_learning=True) for condition in conditions for seed in SEEDS]
        for future in as_completed(pending):
            trial = future.result()
            results.append(trial)
            label = f"{trial['architecture']}_{trial['seed']}"
            save_json(directory / f"audit_{label}.json", trial)
            print(label, trial["first_verified_population"], flush=True)
    prior = json.loads((ROOT / "results" / "transfer_guard" / "verified_matched_comparison.json").read_text())
    summary = {}
    for condition in conditions:
        name = condition.id
        rows = sorted((row for row in results if row["architecture"] == name), key=lambda row: row["seed"])
        first = [row["first_verified_population"] for row in rows]
        verified = [number for number in first if number is not None]
        summary[name] = {
            "seeds": SEEDS,
            "prior_first_finish_population": prior[name]["first_nominal_finish_population"],
            "audited_first_finish_population": first,
            "first_population_finishes": first.count(1),
            "training_runs_with_a_verified_finish": len(verified),
            "median_first_finish_population_among_successes": median(verified) if verified else None,
        }
    save_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
