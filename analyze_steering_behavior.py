"""Probe frozen steering policies and replay the complete initial candidate pools."""

from concurrent.futures import ProcessPoolExecutor
import argparse
from itertools import combinations
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

import pilot
from footprint_guard import weights
from fresh_guard_check import ARCHIVES
from paper_corridors import ROOT, drive, load_corridors, steering_command
from run_experiments import save_json
from source_reproduction import OPTIONS
from transfer_experiment import CONDITIONS, POPULATION, PROJECTED_CONTEXT, SEEDS, assess


OFFSETS = np.linspace(-1.5, 1.5, 31)
WIDTH = 8.0
EARLY_PROGRESS = 0.05
REDUCED_FREE_SCALE = 0.25
INITIAL_ARMS = (
    ("symmetric", 1.0), ("free", 1.0), ("free_small", 1.0), ("projected_context", 1.0),
    ("symmetric", REDUCED_FREE_SCALE), ("free", REDUCED_FREE_SCALE),
)
SWEEP_SCALES = (0.0625, 0.125, 0.25, 0.5, 1.0)


def corridor_scans(offsets):
    """Analytic straight corridor with no front wall within sensor range."""
    sine = np.sin(pilot.BEAM_ANGLES)
    wall = np.where(sine >= 0, WIDTH / 2, -WIDTH / 2)
    ranges = np.divide(wall - offsets[:, None], sine, where=np.abs(sine) > 1e-12,
                       out=np.full((len(offsets), len(sine)), pilot.SENSOR_RANGE))
    return np.minimum(ranges, pilot.SENSOR_RANGE) / pilot.SENSOR_RANGE


def steer(genomes, scans, condition):
    """Return one row of steering commands per genome for all probe scans."""
    return np.asarray([
        np.rad2deg(OPTIONS["steering_scale"] * steering_command(
            scans * OPTIONS["input_scale"],
            *pilot.decode(np.repeat(genome[None], len(scans), axis=0), condition.symmetric, condition.hidden),
            condition.symmetric, condition.projected))
        for genome in genomes
    ])


def track_scans(course):
    indices = np.linspace(8, len(course.route) - 9, 128, dtype=int)
    positions = course.route[indices]
    tangent = course.route[indices + 8] - course.route[indices - 8]
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    return pilot.sense(positions - pilot.HALF_LENGTH * pilot.direction(heading), heading, course)


def frozen_analysis(course):
    road_scans = track_scans(course)
    probes = corridor_scans(OFFSETS)
    np.testing.assert_allclose(probes[::-1], probes[:, ::-1], atol=1e-12)
    records = []
    for architecture, archive in ARCHIVES.items():
        condition = CONDITIONS[architecture]
        genomes = np.vstack([weights(seed, architecture, archive) for seed in SEEDS])
        angles = steer(genomes, probes, condition)
        direct = steer(genomes, road_scans, condition)
        reflected = steer(genomes, road_scans[:, ::-1], condition)
        mirror_error = direct + reflected
        if condition.symmetric:
            np.testing.assert_allclose(mirror_error, 0, atol=1e-10)
            np.testing.assert_allclose(angles[:, len(OFFSETS) // 2], 0, atol=1e-10)
        for seed, response, error, original, mirror in zip(SEEDS, angles, mirror_error, direct, reflected):
            source = ROOT / "results" / archive / f"outcome_{architecture}_restarts_{seed}.json"
            archived = json.loads(source.read_text())["courses"]
            magnitude = np.sqrt(np.mean((original ** 2 + mirror ** 2) / 2))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            records.append({
                "seed": seed, "architecture": architecture, "probe_steering_deg": response.tolist(),
                "mirror_rmse_deg": rmse,
                "command_rmse_deg": float(magnitude),
                "relative_mirror_rmse": rmse / magnitude if magnitude else None,
                "corrects_at_both_outer_offsets": bool(response[0] > 0 and response[-1] < 0),
                "finishes_mirrored_training_course": archived["map6"]["mirror"]["finished"],
                "heldout_finishes": sum(archived[key]["original"]["finished"] for key in ("map7", "map8", "map9")),
            })
    return records


def audit_initial(course, architecture, seed, scale, mirror_probes=True):
    condition = PROJECTED_CONTEXT if architecture == PROJECTED_CONTEXT.id else CONDITIONS[architecture]
    genes = (14 if condition.symmetric else 27) * condition.hidden
    genomes = np.random.default_rng(seed).uniform(-1, 1, (POPULATION, genes)) * scale
    _, raw_finishes, progress, _ = drive(genomes, course=course, **condition.network(), **OPTIONS)
    verified = np.zeros(POPULATION, dtype=bool)
    for index in np.flatnonzero(raw_finishes):
        result = assess(genomes[index:index + 1], course, condition)
        verified[index] = result["finished"]
    start_scan = pilot.sense(
        (course.start - pilot.HALF_LENGTH * pilot.direction(course.heading))[None],
        np.asarray([course.heading]), course,
    )
    scans = np.vstack((corridor_scans(np.array([0.0])), start_scan))
    if mirror_probes:
        road = track_scans(course)
        scans = np.vstack((scans, road, road[:, ::-1]))
    angles = steer(genomes, scans, condition)
    row = {
        "seed": seed, "architecture": architecture, "weight_scale": scale, "candidate_count": POPULATION,
        "verified_finishes": int(verified.sum()), "step_end_finishes": int(raw_finishes.sum()),
        "reached_at_most_five_percent_at_step_end": int(np.count_nonzero(progress <= EARLY_PROGRESS)),
        "median_progress_at_step_end": float(np.median(progress)),
        "verified_finish_by_candidate": verified.tolist(), "progress_at_step_end": progress.tolist(),
        "centred_corridor_command_deg": angles[:, 0].tolist(), "training_start_command_deg": angles[:, 1].tolist(),
    }
    if mirror_probes:
        direct, mirror = np.split(angles[:, 2:], 2, axis=1)
        magnitude = np.sqrt(np.mean((direct ** 2 + mirror ** 2) / 2, axis=1))
        relative = np.divide(np.sqrt(np.mean((direct + mirror) ** 2, axis=1)), magnitude,
                             out=np.zeros_like(magnitude), where=magnitude > 0)
        row["relative_mirror_rmse_by_candidate"] = relative.tolist()
    return row


def scale_sweep(course, destination):
    initial = destination / "initial.json"
    rows = json.loads(initial.read_text())["runs"] if initial.exists() else []
    existing = {(row["architecture"], row["seed"], row["weight_scale"]): row for row in rows}
    cases = [(architecture, seed, scale)
             for architecture in ("symmetric", "free", "free_small", "projected_context")
             for scale in SWEEP_SCALES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {case: pool.submit(audit_initial, course, *case, False)
                for case in cases if case not in existing}
        rows = [existing[case] if case in existing else jobs[case].result() for case in cases]
    save_json(destination / "initial_scale_sweep.json", {"scales": SWEEP_SCALES, "runs": rows})


def plot_responses(rows, destination):
    fig, axes = pilot.plt.subplots(1, 2, figsize=(7.1, 2.65), sharex=True, sharey=True, layout="constrained")
    for ax, architecture, title, colour in zip(axes, ("symmetric", "free"),
                                               ("Symmetric", "Unrestricted"), ("#176983", "#9c4932")):
        subset = [r for r in rows if r["architecture"] == architecture]
        responses = np.array([r["probe_steering_deg"] for r in subset])
        for response in responses:
            ax.plot(OFFSETS, response, color=colour, alpha=0.28, linewidth=0.8)
        ax.plot(OFFSETS, np.median(responses, axis=0), color=colour, linewidth=2)
        ax.axhline(0, color="#727b83", linewidth=0.65)
        ax.axvline(0, color="#727b83", linewidth=0.65)
        ax.set(title=title, xlabel="Lateral position (m)", xticks=[-1.5, 0, 1.5])
    axes[0].set_ylabel("Steering command (deg)")
    fig.savefig(destination, dpi=250)
    pilot.plt.close(fig)


def distinct_rank_sums(ranks, levels, counts, indices):
    """Enumerate each assignment of the tied outcome ranks once."""
    if len(levels) == 1:
        yield levels[0] * ranks[list(indices)].sum()
        return
    for chosen in combinations(indices, counts[0]):
        remaining = tuple(i for i in indices if i not in chosen)
        contribution = levels[0] * ranks[list(chosen)].sum()
        for tail in distinct_rank_sums(ranks, levels[1:], counts[1:], remaining):
            yield contribution + tail


def mirror_permutation(rows):
    """Compare two mirror-error measures against two outcomes for the unrestricted GA runs."""
    free = [row for row in rows if row["architecture"] == "free"]
    associations = []
    for feature in ("mirror_rmse_deg", "relative_mirror_rmse"):
        x = rankdata([row[feature] for row in free])
        x -= x.mean()
        for outcome in ("heldout_finishes", "finishes_mirrored_training_course"):
            y = rankdata([row[outcome] for row in free])
            y -= y.mean()
            levels, counts = np.unique(y, return_counts=True)
            null = np.fromiter(distinct_rank_sums(x, levels, counts, tuple(range(len(free)))), dtype=float)
            extreme = np.count_nonzero(np.abs(null) >= abs(x @ y) - 1e-9)
            associations.append({"feature": feature, "outcome": outcome, "rho": float(spearmanr(x, y).statistic),
                                 "permutations": len(null), "at_least_as_extreme": int(extreme),
                                 "p_two_sided_unadjusted": float(extreme / len(null))})
    return {"networks": len(free), "exploratory": True, "associations": associations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--frozen-only", action="store_true", help="Skip replaying initial generations")
    modes.add_argument("--scale-sweep", action="store_true", help="Compare initial weights across five scales")
    modes.add_argument("--mirror-associations", action="store_true", help="Analyse saved mirror probes with exact permutations")
    args = parser.parse_args()
    destination = ROOT / "results" / "steering_behavior"
    destination.mkdir(exist_ok=True)
    if args.mirror_associations:
        frozen = json.loads((destination / "frozen.json").read_text())["runs"]
        save_json(destination / "mirror_permutation.json", mirror_permutation(frozen))
        return
    course = load_corridors()["map6"]
    if args.scale_sweep:
        scale_sweep(course, destination)
        return
    frozen = frozen_analysis(course)
    save_json(destination / "frozen.json", {"corridor_width_m": WIDTH, "offsets_m": OFFSETS.tolist(), "runs": frozen})
    plot_responses(frozen, destination / "steering_response.png")
    finishes = {(row["architecture"], row["seed"]): row["heldout_finishes"] for row in frozen}
    differences = np.array([finishes["symmetric", seed] - finishes["free", seed] for seed in SEEDS])
    counts = np.bincount(differences + 3, minlength=7)
    bootstrap = np.array([1], dtype=np.int64)
    for _ in SEEDS:
        bootstrap = np.convolve(bootstrap, counts)
    cumulative = np.cumsum(bootstrap) / len(SEEDS) ** len(SEEDS)
    interval = (np.searchsorted(cumulative, [0.025, 0.975]) - 3 * len(SEEDS)) / len(SEEDS)
    save_json(destination / "paired_transfer.json", {
        "seed_pairs": len(SEEDS), "fixed_test_courses": ["map7", "map8", "map9"],
        "differences_by_seed": differences.tolist(), "mean_extra_courses": float(differences.mean()),
        "paired_percentile_bootstrap_95": interval.tolist(),
    })
    if not args.frozen_only:
        jobs = [(course, architecture, seed, scale)
                for architecture, scale in INITIAL_ARMS for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(audit_initial, *job) for job in jobs]
            results = []
            for task in pending:
                row = task.result()
                print(row["architecture"], row["weight_scale"], row["seed"], row["verified_finishes"], flush=True)
                results.append(row)
        save_json(destination / "initial.json", {
            "early_progress_threshold": EARLY_PROGRESS, "runs": results,
            "replay": "reference evaluator with ten sampled body poses for each apparent finish",
        })


if __name__ == "__main__":
    main()
