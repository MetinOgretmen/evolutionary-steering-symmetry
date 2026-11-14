"""Quantify paired transfer and plot a reproducible successful example."""

import json
from itertools import product
from math import comb

import numpy as np

import pilot
from paper_corridors import ROOT, SCALE, drive, load_corridors
from run_experiments import save_json
from source_reproduction import OPTIONS
from transfer_experiment import CONDITIONS, restart_scenarios


TESTS = ("map7", "map8", "map9")


def complete(row):
    return finish_count(row) == len(TESTS)


def finish_count(row):
    return sum(row["courses"][name]["original"]["finished"] for name in TESTS)


def main():
    directory = ROOT / "results" / "transfer"
    courses = load_corridors()
    results = {}
    for path in sorted(directory.glob("outcome_*.json")):
        row = json.loads(path.read_text())
        key = (row["architecture"], row["restarts"])
        results.setdefault(key, {})[row["seed"]] = row

    symmetric = results["symmetric", True]
    paired = {}
    for key, baseline in results.items():
        if baseline is symmetric:
            continue
        pairs = [(complete(row), complete(baseline[seed])) for seed, row in symmetric.items()]
        wins, losses = pairs.count((True, False)), pairs.count((False, True))
        discordant = wins + losses
        p_value = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
        differences = [finish_count(row) - finish_count(baseline[seed]) for seed, row in symmetric.items()]
        absolute = [abs(difference) for difference in differences if difference]
        permutations = [sum(sign * value for sign, value in zip(signs, absolute))
                        for signs in product((-1, 1), repeat=len(absolute))]
        observed = abs(sum(differences))
        paired[f"{key[0]}_{'restarts' if key[1] else 'fixed'}"] = {
            "symmetric_only": wins, "baseline_only": losses, "both": pairs.count((True, True)),
            "neither": pairs.count((False, False)), "paired_exact_two_sided_p": p_value,
            "more_courses_for_symmetric": sum(value > 0 for value in differences),
            "more_courses_for_baseline": sum(value < 0 for value in differences),
            "same_number_of_courses": differences.count(0),
            "course_count_difference_by_seed": dict(zip(symmetric, differences)),
            "course_count_sign_flip_two_sided_p": sum(abs(value) >= observed for value in permutations)
            / len(permutations),
        }

    training = courses["map6"]
    indices = np.linspace(8, len(training.route) - 9, 128, dtype=int)
    poses = training.route[indices]
    directions = training.route[indices + 8] - training.route[indices - 8]
    headings = np.arctan2(directions[:, 1], directions[:, 0])
    scans = pilot.sense(poses - pilot.HALF_LENGTH * pilot.direction(headings), headings,
                       training) * OPTIONS["input_scale"]
    reflection_errors = {}
    for (architecture, restarts), arm in results.items():
        label = f"{architecture}_{'restarts' if restarts else 'fixed'}"
        reflection_errors[label] = {}
        condition = CONDITIONS[architecture]
        for seed in arm:
            archived = json.loads((directory / f"weights_{label}_{seed}.json").read_text())
            genome = np.repeat([archived["champion"]], len(scans), axis=0)
            weights = pilot.decode(genome, condition.symmetric, condition.hidden)
            original = pilot.command(scans, *weights, condition.symmetric)
            reflection = pilot.command(scans[:, ::-1], *weights, condition.symmetric)
            error = np.rad2deg(OPTIONS["steering_scale"] * (original + reflection))
            reflection_errors[label][str(seed)] = float(np.sqrt(np.mean(error**2)))
    save_json(directory / "behavior_and_pairs.json", {"paired_comparisons": paired,
                                                       "steering_reflection_rmse_deg": reflection_errors})
    selection_replay = {}
    for (architecture, restarts), arm in results.items():
        if not restarts:
            continue
        seeds = sorted(arm)
        archives = [json.loads((directory / f"weights_{architecture}_restarts_{seed}.json").read_text())
                    for seed in seeds]
        genomes = np.asarray([archive["champion"] for archive in archives])
        finishes = [drive(genomes, course=scenario, **CONDITIONS[architecture].network(), **OPTIONS)[1]
                    for scenario in [training, *restart_scenarios(training)]]
        selection_replay[architecture] = {
            str(seed): outcomes.tolist() for seed, outcomes in zip(seeds, np.asarray(finishes).T)
        }
    save_json(directory / "selection_replay.json", selection_replay)
    examples = [seed for seed, row in symmetric.items() if complete(row)]
    if examples:
        plot_example(directory, courses, results, min(examples))


def plot_example(directory, courses, results, seed):
    fig, axes = pilot.plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for axis, name, number in zip(axes, TESTS, (11, 13, 15)):
        course = courses[name]
        height, width = course.mask.shape
        axis.imshow(course.mask, extent=(-0.5 * SCALE, (width - 0.5) * SCALE,
                                        -(height - 0.5) * SCALE, 0.5 * SCALE),
                    cmap="Greys_r", alpha=0.28, interpolation="nearest")
        for architecture, color, title in (("symmetric", "#086789", "Symmetric 25–15–2"),
                                           ("free", "#b9581b", "Free 25–15–2"),
                                           ("free_small", "#7a4884", "Free 25–8–2")):
            archive = json.loads((directory / f"weights_{architecture}_restarts_{seed}.json").read_text())
            genome = np.asarray([archive["champion"]])
            _, _, _, path = drive(genome, course=course, trace=True,
                                   **CONDITIONS[architecture].network(), **OPTIONS)
            path = np.asarray(path)
            finished = results[architecture, True][seed]["courses"][name]["original"]["finished"]
            axis.plot(*path.T, lw=1.2, color=color, label=title)
            if not finished:
                axis.scatter(*path[-1], color=color, marker="x", s=38, zorder=4)
        axis.scatter(*course.start, color="#111111", s=18, zorder=5)
        axis.scatter(*course.goal, facecolor="white", edgecolor="#111111", s=24, zorder=5)
        axis.set(title=f"Figure {number} · held out", xlabel="x [m]", ylabel="y [m]")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Same seed {seed} · all controllers trained only on Figure 9")
    fig.savefig(directory / "example_trajectories.png", dpi=180)
    pilot.plt.close(fig)


if __name__ == "__main__":
    main()
