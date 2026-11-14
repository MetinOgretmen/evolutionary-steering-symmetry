"""Replay archived controllers to measure held-out progress and inspect steering."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import pilot
from run_experiments import make_scenarios
from summarize_experiments import LABELS, load_runs


def replay_run(job):
    run, condition, scenarios, noise_seed = job
    genome = np.array([run["champion"]])
    network = {name: condition[name] for name in ("symmetric", "hidden", "physical_beams")}
    by_name = {name: (index, course, start) for index, (name, course, start) in enumerate(scenarios)}
    metrics = []
    for episode in run["test_episodes"]:
        name = episode["scenario"]
        index, course, start = by_name[name]
        sigma = episode["noise_std"]
        noise = None
        if sigma:
            rng = np.random.default_rng([noise_seed, run["seed"], index, episode["repeat"]])
            noise = sigma * rng.standard_normal((pilot.MAX_TICKS, len(pilot.BEAM_ANGLES)))
        score, finished, path = pilot.evaluate(genome, **network, record=True, course=course,
                                               start=start, noise=noise)
        if bool(finished[0]) != episode["finished"] or not np.isclose(score[0], episode["fitness"], atol=1e-10):
            raise ValueError(f"Replay differs from archived outcome for {run['condition']}, {run['seed']}, {name}")
        # The archived fitness is based on the last safe x, while the trace includes a contact pose.
        progress_m = np.sqrt(score[0] * (len(path) - 1))
        safe_x = start[0] + progress_m
        metrics.append({
            "scenario": name,
            "noise_std": sigma,
            "progress": float(progress_m / (course.finish_x - start[0])),
            "safe_x": float(safe_x),
            "past_first": bool(safe_x >= course.obstacles[0, 2]),
            "past_second": bool(safe_x >= course.obstacles[1, 2]),
            "finished": bool(finished[0]),
        })
    return metrics


def summarize_progress(groups, replays, sigmas):
    summary = {}
    for label, runs in groups.items():
        by_seed = [
            [np.mean([episode["progress"] for episode in replay if episode["noise_std"] == sigma])
             for sigma in sigmas]
            for replay in replays[label]
        ]
        clean = [episode for replay in replays[label] for episode in replay if episode["noise_std"] == 0]
        summary[label] = {
            "mean_progress_by_noise": np.mean(by_seed, axis=0).tolist(),
            "seed_progress_by_noise": by_seed,
            "clean_episodes": len(clean),
            "clean_past_first": sum(episode["past_first"] for episode in clean),
            "clean_past_second": sum(episode["past_second"] for episode in clean),
            "clean_finished": sum(episode["finished"] for episode in clean),
            "seeds": [run["seed"] for run in runs],
        }
    differences = np.array(summary["symmetric"]["seed_progress_by_noise"]) - summary["free"]["seed_progress_by_noise"]
    count = len(differences)
    indices = np.random.default_rng(4350).integers(count, size=(10_000, count))
    draws = differences[indices].mean(axis=1)
    return {
        "noise_levels": sigmas,
        "conditions": summary,
        "paired_progress_symmetric_minus_free": {
            "mean_difference_by_noise": differences.mean(axis=0).tolist(),
            "ci95_by_noise": np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
        },
    }


def controller_trace(run, symmetric, course, start):
    genome = np.array([run["champion"]])
    _, _, path = pilot.evaluate(genome, symmetric, record=True, course=course, start=start)
    path = np.array(path)
    position = start[None, :].copy()
    heading = np.zeros(1)
    inputs, outputs = pilot.decode(genome, symmetric)
    signals = []
    for centre in path[1:]:
        scan = pilot.sense(position, heading, course)
        steering = pilot.command(scan, inputs, outputs, symmetric)
        signals.append((float(position[0, 0]), float(np.rad2deg(steering[0])), scan[0] * pilot.SENSOR_RANGE))
        position, heading = pilot.move(position, heading, steering)
        np.testing.assert_allclose(position[0], centre, rtol=0, atol=1e-10)
    x, steering, scans = zip(*signals)
    return path, np.array(x), np.array(steering), np.stack(scans)


def select_example(groups, replays, scenarios):
    # Choose the first pair in seed and scenario order that both pass the first
    # obstacle, then fail before passing the second. This illustrates the common failure.
    for index, free in enumerate(replays["free"]):
        symmetric = replays["symmetric"][index]
        for left, right in zip(free, symmetric):
            if left["noise_std"] != 0:
                continue
            if all(row["past_first"] and not row["past_second"] for row in (left, right)):
                name, course, start = next(case for case in scenarios if case[0] == left["scenario"])
                seed = groups["free"][index]["seed"]
                traces = {
                    label: controller_trace(groups[label][index], label == "symmetric", course, start)
                    for label in ("free", "symmetric")
                }
                return seed, name, course, traces
    raise ValueError("No paired second-obstacle failure in the clean test episodes.")


def plot_behavior(summary, example, output):
    seed, name, course, traces = example
    fig, axes = pilot.plt.subplots(2, 2, figsize=(11.4, 6.7), layout="constrained")
    progress, track, steering, ranges = axes.ravel()
    colors = ("#246c9a", "#a43c21", "#655bb2", "#4a958b", "#bc7c3a", "#545d66", "#9a476e")
    for (label, row), color in zip(summary["conditions"].items(), colors):
        rates = np.array(row["mean_progress_by_noise"]) * 100
        progress.plot(range(len(rates)), rates, marker="o", color=color, label=LABELS[label])
    sigmas = summary["noise_levels"]
    progress.set(title="Held-out progress · 12 GA seeds per condition", xlabel="Relative noise SD",
                 ylabel="Mean course progress [%]", ylim=(0, 100),
                 xticks=range(len(sigmas)), xticklabels=[f"{sigma:.0%}" for sigma in sigmas])
    progress.legend(fontsize=7, ncols=2, loc="upper right")

    for (label, (path, x, delta, scan)), color, style in zip(traces.items(), colors, ("-", "--")):
        track.plot(path[:, 0], path[:, 1], color=color, linestyle=style, label=LABELS[label], linewidth=2)
        track.scatter(*path[-1], color=color, marker="x", s=40, zorder=3)
        steering.plot(x, delta, color=color, linestyle=style, linewidth=2)
        ranges.plot(x, scan[:, 12], color=color, linestyle=style, linewidth=2)
    for x0, y0, x1, y1 in course.obstacles:
        track.add_patch(pilot.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#707982"))
        for axis in (steering, ranges):
            axis.axvspan(x0, x1, color="#707982", alpha=0.14, linewidth=0)
    x_limits = (max(0, course.obstacles[0, 0] - 5), course.obstacles[1, 2] + 3)
    track.set(title=f"{name} · seed {seed} · × contact", xlim=x_limits,
              ylim=(0, pilot.COURSE_WIDTH), aspect="equal", xlabel="Vehicle centre x [m]", ylabel="y [m]")
    track.legend(fontsize=8, loc="lower right")
    steering.axhline(0, color="#555555", linewidth=0.7)
    steering.set(title="Front-wheel command · negative is right", xlim=x_limits,
                 xlabel="Vehicle centre x [m]", ylabel="Steering [°]")
    ranges.set(title="Forward beam · measured by both sensors", xlim=x_limits, ylim=(0, pilot.SENSOR_RANGE),
               xlabel="Vehicle centre x [m]", ylabel="Range [m]")
    for axis in (progress, steering, ranges):
        axis.grid(alpha=0.18)
    fig.savefig(output, dpi=170)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config, groups = load_runs(args.results)
    scenarios = make_scenarios(config["test_courses"], config["geometry"], config["mirror_test_courses"])
    replays = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for condition in config["conditions"]:
            label = condition["id"]
            jobs = [(run, condition, scenarios, config["sensor"]["noise_seed"]) for run in groups[label]]
            replays[label] = list(pool.map(replay_run, jobs))
            print(f"Verified {label}: {sum(map(len, replays[label]))} replays", flush=True)
    summary = summarize_progress(groups, replays, config["sensor"]["test_noise_std"])
    example = select_example(groups, replays, scenarios)
    summary["behavior_example"] = {"seed": example[0], "scenario": example[1]}
    (args.results / "behavior_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_behavior(summary, example, args.results / "behavior.png")
    for label, row in summary["conditions"].items():
        print(label, "progress", [round(value * 100, 1) for value in row["mean_progress_by_noise"]],
              "past first / second", row["clean_past_first"], row["clean_past_second"])


if __name__ == "__main__":
    main()
