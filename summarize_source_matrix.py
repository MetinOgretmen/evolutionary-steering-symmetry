"""Audit and summarize the seven source-aligned single-course conditions."""

import argparse
import json
from pathlib import Path

import numpy as np

import pilot
from paper_corridors import ROOT, drive, load_corridors
from source_reproduction import Condition, OPTIONS
from summarize_experiments import LABELS
from verify_source_controller import interpolated_contacts


def summarize(directory, conditions, seeds):
    courses = load_corridors()
    rows = {}
    for condition in conditions:
        training = []
        first = []
        tests = []
        verified = []
        noisy = []
        for seed in seeds:
            suffix = f"{condition.id}_{seed}.json"
            frozen = json.loads((directory / f"weights_{suffix}").read_text())
            run = json.loads((directory / f"outcome_{suffix}").read_text())
            if any(run[key] != value for key, value in frozen.items()) or "test" in frozen:
                raise ValueError(f"Training archive differs from test record for {suffix}")
            training.append(run["training"]["finished"])
            first.append(run["first_verified_finish_population"])
            clean = [run["test"][name]["finished"] for name in ("map7", "map8", "map9")]
            tests.append(clean)
            genome = np.array([run["champion"]])
            checked = []
            for name, finished in zip(("map7", "map8", "map9"), clean):
                if not finished:
                    checked.append(False)
                    continue
                _, result, _, path = drive(genome, course=courses[name], trace=True,
                                           **condition.network(), **OPTIONS)
                checked.append(bool(result[0]) and not interpolated_contacts(
                    genome, courses[name], np.array(path), **condition.network()))
            verified.append(checked)
            episodes = run["noise_test"]
            levels = [[row["finished"] for row in episodes if row["sigma"] == sigma]
                      for sigma in (0.05, 0.10)]
            if any(len(level) != 9 for level in levels):
                raise ValueError(f"Incomplete noise episodes for {suffix}")
            noisy.append([sum(level) for level in levels])
        clean = np.asarray(tests)
        checked = np.asarray(verified)
        rows[condition.id] = {
            "training_finishes": sum(training),
            "first_verified_training_finish": first,
            "clean_finishes_by_map": clean.sum(axis=0).tolist(),
            "all_three_clean": int(clean.all(axis=1).sum()),
            "interpolated_clean_finishes_by_map": checked.sum(axis=0).tolist(),
            "all_three_interpolated_clean": int(checked.all(axis=1).sum()),
            "noise_finishes_at_5_and_10_percent": np.sum(noisy, axis=0).tolist(),
        }
        print(condition.id, rows[condition.id], flush=True)
    return {"seeds": seeds, "test_maps": ["map7", "map8", "map9"], "conditions": rows}


def plot(summary, output):
    count = len(summary["seeds"])
    conditions = summary["conditions"]
    fig, (clean, noise) = pilot.plt.subplots(1, 2, figsize=(10.2, 4.8), layout="constrained",
                                            gridspec_kw={"width_ratios": (1.5, 1)})
    names = [LABELS[name] for name in conditions]
    main = np.array([[row["training_finishes"], *row["clean_finishes_by_map"], row["all_three_clean"]]
                     for row in conditions.values()])
    noisy = np.array([row["noise_finishes_at_5_and_10_percent"] for row in conditions.values()])
    for axis, values, maximum, labels, title in [
        (clean, main, count, ["Train\nFig. 9", "Fig. 11", "Fig. 13", "Fig. 15", "All tests"], "Clean range data"),
        (noise, noisy, count * 9, ["5% SD", "10% SD"], "Noisy test episodes"),
    ]:
        axis.imshow(values / maximum, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(names)), title=title)
        axis.set_yticklabels(names if axis is clean else [""] * len(names))
        for row, col in np.ndindex(values.shape):
            color = "white" if values[row, col] / maximum >= 0.58 else "#17283a"
            axis.text(col, row, f"{values[row, col]}/{maximum}", color=color, ha="center", va="center")
    fig.suptitle("Source-aligned comparison · training and selection on Figure 9 only")
    fig.savefig(output, dpi=175)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results" / "source_matrix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6200, 6201, 6202])
    args = parser.parse_args()
    config = json.loads((ROOT / "experiment_config.json").read_text())
    conditions = [Condition(**row) for row in config["conditions"]]
    summary = summarize(args.directory, conditions, args.seeds)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, args.directory / "comparison.png")


if __name__ == "__main__":
    main()
