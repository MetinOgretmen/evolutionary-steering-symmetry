"""Compare the archived clean and noisy training arms and audit test finishes."""

from collections import Counter
import json
from statistics import mean

import numpy as np

import pilot
from paper_corridors import ROOT, drive, load_corridors
from run_experiments import save_json
from source_reproduction import OPTIONS, Condition, test_noise
from verify_source_controller import interpolated_contacts


def read(path):
    return json.loads(path.read_text())


def summarize(directory, prefix, courses):
    clean = []
    noisy = {sigma: {} for sigma in (0.05, 0.10, 0.20)}
    contacts = []
    for seed in (6200, 6201, 6202):
        label = f"{prefix}{seed}"
        trial = read(directory / f"outcome_{label}.json")
        frozen = read(directory / f"weights_{label}.json")
        if frozen["champion"] != trial["champion"] or "test" in frozen:
            raise ValueError(f"Test outcomes do not match the training archive for {label}")
        genome = np.asarray([frozen["champion"]])
        clean.extend(trial["test"].values())
        rows = trial["noise_test"]
        if not any(row["sigma"] == 0.20 for row in rows):
            rows = [*rows, *test_noise(frozen, courses, Condition(), (0.20,))]
        for row in rows:
            noisy[row["sigma"]][seed, row["course"], row["repeat"]] = row
        for index, (name, course) in enumerate(courses.items()):
            episodes = [{"sigma": 0.0, **trial["test"][name]},
                        *(row for row in rows if row["course"] == name)]
            for episode in episodes:
                if not episode["finished"]:
                    continue
                sigma = episode["sigma"]
                noise = None
                if sigma:
                    rng = np.random.default_rng([7400, seed, index, episode["repeat"]])
                    noise = sigma * rng.standard_normal((course.max_ticks, len(pilot.BEAM_ANGLES)))
                _, finished, _, path = drive(genome, True, course, trace=True, noise=noise, **OPTIONS)
                if not finished[0]:
                    raise ValueError(f"Test finish did not replay for {label}, {name}, {episode}")
                count = interpolated_contacts(genome, course, np.asarray(path), noise=noise)
                contacts.append({"seed": seed, "course": name, "sigma": sigma,
                                 "repeat": episode.get("repeat"), "interpolated_contact_poses": count})
    contact_counts = Counter(row["sigma"] for row in contacts if row["interpolated_contact_poses"])
    totals = {}
    for label, rows in [("clean", clean), *((str(s), r.values()) for s, r in noisy.items())]:
        finishes = sum(row["finished"] for row in rows)
        totals[label] = {"finishes": finishes, "episodes": len(rows),
                         "audited_finishes": finishes - contact_counts[0.0 if label == "clean" else float(label)],
                         "mean_progress": mean(row["progress"] for row in rows)}
    pairs = Counter((row["finished"], noisy[0.10][key]["finished"]) for key, row in noisy[0.05].items())
    totals["paired_5_vs_10_percent"] = {
        "both": pairs[True, True], "only_5": pairs[True, False],
        "only_10": pairs[False, True], "neither": pairs[False, False],
    }
    totals["interpolated_contact_check"] = contacts
    return totals


def main():
    courses = load_corridors()
    courses.pop("map6")
    directory = ROOT / "results"
    output = directory / "source_noise_training"
    results = {
        "clean_training": summarize(directory / "source_matrix", "symmetric_", courses),
        "five_percent_training": summarize(output, "", courses),
    }
    save_json(output / "summary.json", results)
    for arm, result in results.items():
        contacts = result["interpolated_contact_check"]
        print(arm, {k: v for k, v in result.items() if k != "interpolated_contact_check"}, flush=True)
        print("Finishes with an intermediate contact:", [r for r in contacts if r["interpolated_contact_poses"]],
              flush=True)


if __name__ == "__main__":
    main()
