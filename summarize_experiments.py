"""Summarize AE4350 results using each GA run as one independent observation."""

import argparse
import json
from pathlib import Path

import numpy as np

from pilot import plt


LABELS = {
    "free": "Free",
    "symmetric": "Symmetric",
    "free_small": "Free, 8 hidden",
    "free_low_resolution": "Free, 7 beams",
    "symmetric_low_resolution": "Symmetric, 7 beams",
    "free_elitism": "Free, 10 elites",
    "symmetric_elitism": "Symmetric, 10 elites",
}


def load_runs(directory):
    config = json.loads((directory / "manifest.json").read_text())["config"]
    groups = {}
    for condition in config["conditions"]:
        label = condition["id"]
        runs = []
        for seed in config["ga"]["seeds"]:
            path = directory / f"{label}_{seed}.json"
            run = json.loads(path.read_text())
            if run["condition"] != label or run["seed"] != seed:
                raise ValueError(f"Run does not match its filename: {path.name}")
            runs.append(run)
        groups[label] = runs
    return config, groups


def test_rates(runs, sigmas, scenario_count, noisy_repeats):
    rates = np.empty((len(runs), len(sigmas)))
    for seed_index, run in enumerate(runs):
        for noise_index, sigma in enumerate(sigmas):
            episodes = [row for row in run["test_episodes"] if row["noise_std"] == sigma]
            expected = scenario_count * (noisy_repeats if sigma else 1)
            keys = {(row["scenario"], row["repeat"]) for row in episodes}
            if len(episodes) != expected or len(keys) != expected:
                raise ValueError(f'Invalid test episodes for {run["condition"]}, seed {run["seed"]}')
            rates[seed_index, noise_index] = np.mean([row["finished"] for row in episodes])
    return rates


def summarize(config, groups):
    sigmas = config["sensor"]["test_noise_std"]
    scenario_count = len(config["test_courses"]) * len(config["geometry"]["start_y_m"])
    if config["mirror_test_courses"]:
        scenario_count *= 2
    population = config["ga"]["population"]
    rates, summary = {}, {}
    for label, runs in groups.items():
        rates[label] = test_rates(runs, sigmas, scenario_count, config["sensor"]["noisy_repeats"])
        history = np.array([run["fitness_history"] for run in runs])
        first = [run["first_full_success_population"] for run in runs]
        summary[label] = {
            "initial_full_success_rate": np.mean([run["initial_full_successes"] for run in runs]) / population,
            "initial_populations_with_full_success": first.count(1),
            "runs_with_full_success": sum(value is not None for value in first),
            "first_full_success_populations": first,
            "median_initial_fitness": float(np.median(history[:, 0])),
            "median_final_fitness": float(np.median(history[:, -1])),
            "mean_test_success_by_noise": rates[label].mean(axis=0).tolist(),
            "seed_test_success_by_noise": rates[label].tolist(),
        }
    contrasts = {
        "symmetry": rates["symmetric"] - rates["free"],
        "symmetric_vs_free_small": rates["symmetric"] - rates["free_small"],
        "resolution_free_7_vs_25": rates["free_low_resolution"] - rates["free"],
        "resolution_symmetric_7_vs_25": rates["symmetric_low_resolution"] - rates["symmetric"],
        "elitism_free_10_vs_0": rates["free_elitism"] - rates["free"],
        "elitism_symmetric_10_vs_0": rates["symmetric_elitism"] - rates["symmetric"],
    }
    n_seeds = len(config["ga"]["seeds"])
    indices = np.random.default_rng(4350).integers(n_seeds, size=(10_000, n_seeds))
    paired = {}
    for label, values in contrasts.items():
        draws = values[indices].mean(axis=1)
        paired[label] = {
            "mean_difference_by_noise": values.mean(axis=0).tolist(),
            "ci95_by_noise": np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
        }
    return {"noise_levels": sigmas, "conditions": summary, "paired_test_contrasts": paired}


def plot_overview(config, groups, summary, path):
    fig, (learning, success, generalization) = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    generations = np.arange(1, config["ga"]["populations_evaluated"] + 1)
    evaluated = config["ga"]["population"] * generations
    colors = ("#246c9a", "#a43c21", "#655bb2", "#4a958b", "#bc7c3a", "#545d66", "#9a476e")
    styles = ("-", "-", "--", "--", "--", ":", ":")
    for (label, runs), color, style in zip(groups.items(), colors, styles):
        history = np.array([run["fitness_history"] for run in runs])
        learning.plot(evaluated, np.median(history, axis=0), color=color, linestyle=style, label=LABELS[label])
        first = np.array([run["first_full_success_population"] or np.inf for run in runs])
        reached = np.mean(first[:, None] <= generations, axis=0)
        success.step(evaluated, reached * 100, where="post", color=color, linestyle=style)
        rates = np.array(summary["conditions"][label]["mean_test_success_by_noise"]) * 100
        generalization.plot(np.arange(len(rates)), rates, marker="o", color=color, linestyle=style)
    learning.set(title="Training fitness", xlabel="Candidates evaluated", ylabel="Median best fitness [m²/tick]")
    success.set(title="All four training cases solved", xlabel="Candidates evaluated", ylabel="GA runs [%]")
    sigmas = config["sensor"]["test_noise_std"]
    generalization.set(title="Unseen test courses", xlabel="Relative noise SD", ylabel="Mean completion [%]",
                       xticks=np.arange(len(sigmas)), xticklabels=[f"{sigma:.0%}" for sigma in sigmas])
    for axis in (learning, success, generalization):
        axis.grid(alpha=0.18)
    success.set_ylim(-3, 103)
    generalization.set_ylim(-3, 103)
    fig.legend(*learning.get_legend_handles_labels(), loc="outside lower center", ncols=4, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    config, groups = load_runs(args.results)
    summary = summarize(config, groups)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_overview(config, groups, summary, args.results / "overview.png")
    for label, stats in summary["conditions"].items():
        rates = ", ".join(f"{rate:.1%}" for rate in stats["mean_test_success_by_noise"])
        print(f"{label}: test success {rates}")


if __name__ == "__main__":
    main()
