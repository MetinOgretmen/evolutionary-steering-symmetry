"""Extend the frozen initial-pool sweep and isolate the effect of mirror projection."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.stats import binomtest

from analyze_steering_behavior import audit_initial, steer, track_scans
from paper_corridors import ROOT, load_corridors
from run_experiments import save_json
from transfer_experiment import CONDITIONS, POPULATION, PROJECTED_CONTEXT, SEEDS

ARCHITECTURES = ("symmetric", "free", "free_small", "projected_context")
SOURCE = ROOT / "results" / "steering_behavior"
DESTINATION = ROOT / "results" / "followup_initial"


def run_sweep(course, pool):
    rows = json.loads((SOURCE / "initial_scale_sweep.json").read_text())["runs"]
    cached_path = DESTINATION / "sweep_replays.json"
    cached = json.loads(cached_path.read_text())["runs"] if cached_path.exists() else []
    cases = {(r["architecture"], r["weight_scale"], r["seed"]): r for r in [*rows, *cached]}

    def add_scales(scales):
        keys = [(arch, scale, seed) for arch in ARCHITECTURES for scale in scales for seed in SEEDS]
        pending = {key: pool.submit(audit_initial, course, key[0], key[2], key[1], False)
                   for key in keys if key not in cases}
        for key, task in pending.items():
            cases[key] = task.result()
        save_json(cached_path, {"runs": [r for key, r in cases.items() if key[1] >= 2]})

    scales = sorted({r["weight_scale"] for r in rows} | {2.0, 4.0})
    add_scales((2.0, 4.0))

    def count(arch, scale):
        return sum(cases[arch, scale, seed]["verified_finishes"] for seed in SEEDS)

    edge = [arch for arch in ARCHITECTURES
            if count(arch, 4) > 0 and count(arch, 4) == max(count(arch, s) for s in scales)]
    if edge:
        scales.append(8.0)
        add_scales((8.0,))
    records = []
    for arch in ARCHITECTURES:
        for scale in scales:
            counts = [cases[arch, scale, seed]["verified_finishes"] for seed in SEEDS]
            records.append({"architecture": arch, "scale": scale, "candidates": POPULATION * len(SEEDS),
                            "finishes": sum(counts), "pools_with_finish": sum(c > 0 for c in counts),
                            **dict(zip((f"seed_{s}" for s in SEEDS), counts, strict=True))})
    return {"triggered_scale_8": edge, "runs": records}


def paired_probe(course, seed, baseline, projected):
    genomes = np.random.default_rng(seed).uniform(-1, 1, (POPULATION, 405))
    scans = track_scans(course)
    commands = steer(genomes, np.vstack((scans, scans[:, ::-1])), CONDITIONS["free"])
    direct, mirrored = np.split(commands, 2, axis=1)
    even = (direct + mirrored) / 2
    odd = (direct - mirrored) / 2
    actual_projected = steer(genomes, scans, PROJECTED_CONTEXT)
    np.testing.assert_allclose(odd, actual_projected, atol=1e-10, rtol=0)
    even_ms, odd_ms = np.mean(even**2, axis=1), np.mean(odd**2, axis=1)
    raw = baseline["verified_finish_by_candidate"]
    new = projected["verified_finish_by_candidate"]
    return [{"seed": seed, "candidate_index": i, "raw_finish": raw[i], "projected_finish": new[i],
             "invariant_energy_share": float(even_ms[i] / (even_ms[i] + odd_ms[i]))
             if even_ms[i] + odd_ms[i] > 0 else None,
             "raw_command_rms_deg": float(np.sqrt(even_ms[i] + odd_ms[i])),
             "invariant_command_rms_deg": float(np.sqrt(even_ms[i])),
             "antisymmetric_command_rms_deg": float(np.sqrt(odd_ms[i])),
             "raw_start_command_deg": baseline["training_start_command_deg"][i],
             "projected_start_command_deg": projected["training_start_command_deg"][i],
             "raw_progress": baseline["progress_at_step_end"][i]}
            for i in range(POPULATION)]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(rows):
    groups = {}
    for raw, projected in ((False, False), (False, True), (True, False), (True, True)):
        members = [r for r in rows if (r["raw_finish"], r["projected_finish"]) == (raw, projected)]
        group = {"count": len(members)}
        if members:
            for key in ("invariant_energy_share", "raw_command_rms_deg", "antisymmetric_command_rms_deg",
                        "raw_start_command_deg", "projected_start_command_deg", "raw_progress"):
                values = [r[key] for r in members if r[key] is not None]
                if values:
                    group[key + "_quartiles"] = np.quantile(np.abs(values), [0.25, 0.5, 0.75]).tolist()
        groups[f"raw_{int(raw)}_projected_{int(projected)}"] = group
    gains, losses = groups["raw_0_projected_1"]["count"], groups["raw_1_projected_0"]["count"]
    return {"groups": groups, "exact_mcnemar_p": float(binomtest(gains, gains + losses).pvalue)
            if gains + losses else 1.0,
            "by_pool": [{"seed": seed,
                         "gains": sum(r["seed"] == seed and not r["raw_finish"] and r["projected_finish"] for r in rows),
                         "losses": sum(r["seed"] == seed and r["raw_finish"] and not r["projected_finish"] for r in rows)}
                        for seed in SEEDS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    DESTINATION.mkdir(exist_ok=True)
    course = load_corridors()["map6"]
    initial = json.loads((SOURCE / "initial.json").read_text())["runs"]
    initial = {(r["architecture"], r["seed"]): r for r in initial if r["weight_scale"] == 1.0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(paired_probe, course, seed, initial["free", seed],
                            initial["projected_context", seed]) for seed in SEEDS]
        paired = [row for job in jobs for row in job.result()]
        write_csv(DESTINATION / "paired_projection_candidates.csv", paired)
        save_json(DESTINATION / "paired_projection_summary.json", paired_summary(paired))
        print("Paired projection complete", flush=True)
        sweep = run_sweep(course, pool)
    write_csv(DESTINATION / "scale_sweep.csv", sweep["runs"])
    save_json(DESTINATION / "scale_sweep.json", sweep)
    print("Scale sweep complete", flush=True)


if __name__ == "__main__":
    main()
