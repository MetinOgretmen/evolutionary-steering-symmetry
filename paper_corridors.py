"""Approximate Eraqi et al. (2022), Figures 9, 11, 13, and 15, from the PDF images."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt, label
from skimage.graph import MCP_Geometric

import pilot


ROOT = Path(__file__).resolve().parent
MASKS = ROOT / "paper_track_masks.npz"
SCALE = 0.5  # Metres per figure pixel, a reconstruction assumption.
SPECS = {
    "map6": (5, "Im9.png", (191, 168), (324, 372)),
    "map7": (5, "Im11.png", (12, 78), (58, 11)),
    "map8": (6, "Im13.png", (129, 193), (556, 285)),
    "map9": (6, "Im15.png", (174, 352), (106, 39)),
}


def digitize(pdf):
    from pypdf import PdfReader

    doc = PdfReader(pdf)
    masks = {}
    for name, (page, image_name, (start_x, start_y), (goal_x, goal_y)) in SPECS.items():
        image = next(image for image in doc.pages[page - 1].images if image.name == image_name)
        rgb = np.asarray(image.image.convert("RGB"))
        barrier = ((rgb == [0, 0, 255]).all(axis=-1)
                   | (rgb == [0, 0, 127]).all(axis=-1)
                   | (rgb == [0, 0, 0]).all(axis=-1))
        barrier = np.pad(barrier, 2)
        if name == "map9":
            barrier = binary_dilation(barrier)
        regions, _ = label(binary_fill_holes(barrier) & ~barrier)
        region = regions[start_y + 2, start_x + 2]
        if not region or regions[goal_y + 2, goal_x + 2] != region:
            raise ValueError(f"Could not recover a connected corridor for {name}")
        masks[name] = (regions == region)[2:-2, 2:-2]
    np.savez_compressed(MASKS, **masks)


def intervals(values):
    changes = np.flatnonzero(np.diff(np.r_[False, values, False]))
    return zip(changes[::2], changes[1::2])


def boundary_boxes(mask):
    vertical = np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1)
    horizontal = np.diff(np.pad(mask, ((1, 1), (0, 0))), axis=0)
    boxes = [
        [x - 0.5, -(high - 0.5), x - 0.5, -(low - 0.5)]
        for x, column in enumerate(vertical.T) for low, high in intervals(column)
    ] + [
        [low - 0.5, -(y - 0.5), high - 0.5, -(y - 0.5)]
        for y, row in enumerate(horizontal) for low, high in intervals(row)
    ]
    return np.asarray(boxes) * SCALE


@dataclass
class Corridor:
    mask: np.ndarray
    start_pixel: tuple[int, int]
    goal_pixel: tuple[int, int]

    def __post_init__(self):
        self.boxes = boundary_boxes(self.mask)
        self.centers = (self.boxes[:, :2] + self.boxes[:, 2:]) / 2
        self.halves = (self.boxes[:, 2:] - self.boxes[:, :2]) / 2
        self.vertical = pilot.axis_segments(self.boxes, 0)
        self.horizontal = pilot.axis_segments(self.boxes, 1)
        self.start = np.asarray(self.start_pixel) * [SCALE, -SCALE]
        self.goal = np.asarray(self.goal_pixel) * [SCALE, -SCALE]
        start, goal = self.start_pixel[::-1], self.goal_pixel[::-1]
        clearance = distance_transform_edt(self.mask)
        costs = np.where(self.mask, 1 + 70 / np.maximum(clearance, 1)**2, np.inf)
        planner = MCP_Geometric(costs)
        planner.find_costs([start], [goal])
        route = np.asarray(planner.traceback(goal))
        self.route = route[:, ::-1] * [SCALE, -SCALE]
        direction = self.route[8] - self.route[0]
        self.heading = np.arctan2(direction[1], direction[0])
        distance = MCP_Geometric(np.where(self.mask, 1.0, np.inf))
        self.distance_to_goal = distance.find_costs([goal])[0] * SCALE
        self.length = float(self.distance_to_goal[start])
        self.max_ticks = int(np.ceil(1.7 * self.length / pilot.STEP_DISTANCE))


def reference(course, lookahead):
    position = course.start[None, :].copy()
    heading = np.array([course.heading])
    path = [position[0].copy()]
    nearest = 0
    for _ in range(course.max_ticks):
        window = course.route[nearest:nearest + 24]
        nearest += int(np.argmin(np.linalg.norm(window - position[0], axis=1)))
        candidates = course.route[nearest:]
        distances = np.linalg.norm(candidates - position[0], axis=1)
        ahead = np.flatnonzero(distances >= lookahead)
        target = candidates[ahead[0]] if len(ahead) else candidates[-1]
        offset = target - position[0]
        target_heading = np.arctan2(offset[1], offset[0])
        error = np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading))
        steering = np.clip(1.5 * error, -pilot.STEERING_LIMIT, pilot.STEERING_LIMIT)
        position, heading = pilot.move(position, heading, steering)
        path.append(position[0].copy())
        if pilot.collides(position, heading, course)[0]:
            break
        if np.linalg.norm(position[0] - course.goal) <= 3.0:
            return True, np.array(path)
    return False, np.array(path)


@dataclass(frozen=True)
class VehicleSetup:
    step_distance: float = pilot.STEP_DISTANCE
    sensor_offset: float = pilot.HALF_LENGTH
    body_offset: float = 0.0


MATLAB_VEHICLE = VehicleSetup(step_distance=1.0, sensor_offset=0.0, body_offset=-pilot.HALF_LENGTH)


def move_vehicle(position, heading, delta, step):
    if np.ndim(step) == 0 and step == pilot.STEP_DISTANCE:
        return pilot.move(position, heading, delta)
    if np.ndim(step):
        step = np.asarray(step)[:, None]
    forward = pilot.direction(heading)
    rear = position - (pilot.WHEELBASE / 2) * forward + step * forward
    front = position + (pilot.WHEELBASE / 2) * forward + step * pilot.direction(heading + delta)
    axle = front - rear
    return (front + rear) / 2, np.arctan2(axle[:, 1], axle[:, 0])


def steering_command(scan, inputs, outputs, symmetric, projected=False):
    original = pilot.command(scan, inputs, outputs, symmetric)
    if not projected:
        return original
    reflected = pilot.command(scan[:, ::-1], inputs, outputs, symmetric)
    return (original - reflected) / 2


def safe_step(step, scan, steering):
    """Use the forward ray and demanded turn to choose 40–100% of the nominal step."""
    straight_clearance = scan[:, 12] * pilot.SENSOR_RANGE
    factor = np.minimum(straight_clearance / (2 * pilot.WHEELBASE), 1 - np.abs(steering) / (np.pi / 3))
    return step * np.clip(factor, 0.4, 1.0)


def drive(genomes, symmetric, course, *, steering_scale=1.0, input_scale=1.0, trace=False,
          physical_beams=25, noise=None, vehicle=VehicleSetup(), hidden=pilot.HIDDEN,
          steering_cap=None, steering_log=None, projected=False, adaptive_speed=False):
    count = len(genomes)
    if steering_log is not None and count != 1:
        raise ValueError("A steering log requires one controller.")
    inputs, outputs = pilot.decode(genomes, symmetric, hidden)
    position = np.tile(course.start, (count, 1))
    heading = np.full(count, course.heading)
    ticks = np.zeros(count, dtype=int)
    finished = np.zeros(count, dtype=bool)
    active = np.arange(count)
    path = [course.start.copy()] if trace else None
    for tick in range(course.max_ticks):
        sensor_position = position[active]
        if vehicle.sensor_offset != pilot.HALF_LENGTH:
            sensor_position = sensor_position + (vehicle.sensor_offset - pilot.HALF_LENGTH) * pilot.direction(heading[active])
        scan = pilot.sense(sensor_position, heading[active], course, physical_beams=physical_beams,
                          relative_noise=None if noise is None else noise[tick])
        delta = steering_scale * steering_command(scan * input_scale, inputs[active], outputs[active],
                                                  symmetric, projected)
        if steering_cap is not None:
            delta = np.clip(delta, -steering_cap, steering_cap)
        step = safe_step(vehicle.step_distance, scan, delta) if adaptive_speed else vehicle.step_distance
        next_position, next_heading = move_vehicle(position[active], heading[active], delta, step)
        body_position = next_position
        if vehicle.body_offset:
            body_position = next_position + vehicle.body_offset * pilot.direction(next_heading)
        collision = pilot.collides(body_position, next_heading, course)
        ticks[active] += 1
        safe = active[~collision]
        position[safe] = next_position[~collision]
        heading[safe] = next_heading[~collision]
        finished[safe] = np.linalg.norm(position[safe] - course.goal, axis=1) <= 3.0
        if trace:
            path.append(next_position[0].copy())
        if steering_log is not None:
            steering_log.append((float(delta[0]), bool(collision[0])))
        active = safe[~finished[safe]]
        if not len(active):
            break
    pixels = np.rint(position / [SCALE, -SCALE]).astype(int)
    remaining = course.distance_to_goal[pixels[:, 1], pixels[:, 0]]
    progress = np.clip(course.length - remaining, 0, course.length)
    progress[finished] = course.length
    return pilot.fitness(progress, ticks), finished, progress / course.length, path


def load_corridors():
    with np.load(MASKS) as masks:
        return {
            name: Corridor(masks[name], spec[2], spec[3])
            for name, spec in SPECS.items()
        }


def train_on_map6(corridors, symmetric, mode, generations, seed, training_margin_px=0):
    course = corridors["map6"]
    if training_margin_px:
        course = Corridor(binary_erosion(course.mask, iterations=training_margin_px),
                          course.start_pixel, course.goal_pixel)
        if not reference(course, 5.0)[0]:
            raise ValueError("The map-aware reference could not complete the margin training course.")
    # The old public code uses a 90-degree multiplier. The v2 paper shows raw inputs.
    scales = {"pilot": (1.0, 1.0), "source_inspired": (90 / 35, pilot.SENSOR_RANGE / SCALE)}
    steering_scale, input_scale = scales[mode]
    options = {"steering_scale": steering_scale, "input_scale": input_scale}
    rng = np.random.default_rng(seed)
    genomes = rng.uniform(-1, 1, (pilot.POPULATION, 210 if symmetric else 405))
    best_score = -np.inf
    first_finish = None
    history = []
    for number in range(1, generations + 1):
        scores, finished, progress, _ = drive(genomes, symmetric, course, **options)
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            champion = genomes[winner:winner + 1].copy()
        history.append(best_score)
        if first_finish is None and finished.any():
            first_finish = number
        if number % 5 == 0 or number == 1:
            print(f"{mode} {'symmetric' if symmetric else 'free'} seed {seed} {number}/{generations}: "
                  f"max progress {progress.max():.1%}; first finish {first_finish}", flush=True)
        if number < generations:
            genomes = pilot.offspring(genomes, scores, rng)
    return {
        "condition": "symmetric" if symmetric else "free",
        "mode": mode,
        "seed": seed,
        "training_margin_m": training_margin_px * SCALE,
        "first_training_finish": first_finish,
        "training_history": history,
        "champion": champion[0].tolist(),
        "courses": {
            name: {
                "finished": bool(finished[0]),
                "progress": float(progress[0]),
                "ticks": len(path) - 1,
            }
            for name, geometry in corridors.items()
            for _, finished, progress, path in [drive(champion, symmetric, geometry, trace=True, **options)]
        },
    }


def plot_geometry(corridors, references, output, trial_paths=None, development=False):
    fig, axes = pilot.plt.subplots(2, 2, figsize=(8.6, 9.2), layout="constrained")
    figures = ("9", "11", "13", "15")
    roles = ("development",) * 4 if development else ("training", "test", "test", "test")
    for axis, (name, course), figure, role in zip(axes.ravel(), corridors.items(), figures, roles):
        height, width = course.mask.shape
        axis.imshow(course.mask, extent=(-0.5 * SCALE, (width - 0.5) * SCALE,
                                         -(height - 0.5) * SCALE, 0.5 * SCALE),
                    cmap="Greys_r", alpha=0.32, interpolation="nearest")
        if references is not None:
            reference_path = references[name]
            axis.plot(reference_path[:, 0], reference_path[:, 1], color="#266b88", lw=1)
        if trial_paths is not None:
            finished, path = trial_paths[name]
            axis.plot(path[:, 0], path[:, 1], color="#ba4e22", lw=1.25)
            axis.scatter(*path[-1], color="#ba4e22", marker="o" if finished else "x", s=32, zorder=4)
            axis.text(0.03, 0.03, "Finished" if finished else "Stopped before finish",
                      transform=axis.transAxes, fontsize=8, backgroundcolor="white")
        axis.scatter(*course.start, color="#19804a", s=20, zorder=3)
        axis.scatter(*course.goal, color="#bc4b36", s=20, zorder=3)
        axis.set(title=f"Map {name[-1]} · Figure {figure} · {role}", xlabel="x [m]", ylabel="y [m]")
    title = "Reconstructed paper corridors"
    if references is not None:
        title += " · blue: map-aware reference"
    if trial_paths is not None:
        title += "\nOrange: evolved symmetric controller"
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    pilot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digitize", type=Path, help="Rebuild masks from the arXiv v2 PDF")
    parser.add_argument("--train", type=int, metavar="POPULATIONS", help="Train on reconstructed map 6")
    parser.add_argument("--seed", type=int, default=4350)
    parser.add_argument("--replicate", type=int, nargs="+", metavar="SEED",
                        help="Repeat a source-informed condition")
    parser.add_argument("--architecture", choices=("symmetric", "free"), default="symmetric",
                        help="Architecture to use with --replicate")
    parser.add_argument("--margin-px", type=int, choices=(0, 1), default=0,
                        help="Add one pixel of obstacle and wall clearance during training only")
    parser.add_argument("--replay", type=Path, help="Verify an archived raw-input symmetric controller")
    args = parser.parse_args()
    if args.train is not None and args.train < 1:
        parser.error("--train must be at least 1")
    if args.replicate and args.train is None:
        parser.error("--replicate requires --train")
    if args.margin_px and not args.replicate:
        parser.error("--margin-px requires --replicate")
    if args.replay and args.train:
        parser.error("--replay and --train are separate operations")
    if args.digitize:
        digitize(args.digitize)
    corridors = load_corridors()
    if args.replay:
        trial = json.loads(args.replay.read_text())
        references = {}
        for name, course in corridors.items():
            success, path = reference(course, 5.0)
            if not success:
                raise ValueError(f"Could not reconstruct reference path for {name}")
            references[name] = path
        plot_trial(corridors, references, trial, args.replay.with_suffix(".png"))
        for name, outcome in trial["courses"].items():
            print(f"{name}: {outcome['progress']:.1%} progress, "
                  f"finished={outcome['finished']}, {outcome['ticks']} ticks")
        return
    results = ROOT / "results"
    reconstruction = {}
    references = {}
    for name, course in corridors.items():
        trials = [(lookahead, *reference(course, lookahead)) for lookahead in (5.0, 4.0, 6.0, 3.0)]
        successful = next((trial for trial in trials if trial[1]), None)
        if successful is None:
            raise ValueError(f"Map-aware reference could not verify {name}")
        lookahead, _, reference_path = successful
        references[name] = reference_path
        reconstruction[name] = {"course_length_m": course.length, "reference_ticks": len(reference_path) - 1,
                                "reference_lookahead_m": lookahead, "archived_controllers": {}}
        print(name, f"length={course.length:.1f} m",
              f"reference={len(reference_path) - 1} ticks at {lookahead} m", flush=True)
        for symmetric, condition in [(False, "free"), (True, "symmetric")]:
            genomes = [
                json.loads((results / f"{condition}_{seed}.json").read_text())["champion"]
                for seed in range(5100, 5112)
            ]
            _, finished, progress, _ = drive(np.asarray(genomes), symmetric, course)
            reconstruction[name]["archived_controllers"][condition] = {
                "finish_count": int(finished.sum()), "n": len(finished), "progress": progress.tolist(),
            }
            print(f"  {condition}: {finished.sum()}/{len(finished)} finished; "
                  f"median progress {np.median(progress):.1%}", flush=True)
    (results / "paper_corridors.json").write_text(json.dumps(reconstruction, indent=2) + "\n")
    plot_geometry(corridors, references, results / "paper_corridors.png")
    if args.train:
        jobs = [(args.architecture == "symmetric", "source_inspired", seed)
                for seed in args.replicate] if args.replicate else [
            (symmetric, mode, args.seed)
            for mode in ("pilot", "source_inspired") for symmetric in (False, True)
        ]
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(train_on_map6, corridors, symmetric, mode, args.train, seed, args.margin_px)
                for symmetric, mode, seed in jobs
            ]
            for future in as_completed(futures):
                trial = future.result()
                key = f"{trial['mode']}_{trial['condition']}_{trial['seed']}"
                if args.margin_px:
                    key += f"_margin{args.margin_px}px"
                (results / f"paper_trial_{key}.json").write_text(json.dumps(trial, indent=2) + "\n")
                print(key, trial["courses"], flush=True)
                if trial["mode"] == "source_inspired" and trial["condition"] == "symmetric" and not args.replicate:
                    plot_trial(corridors, references, trial, results / f"paper_trial_{key}.png")


def plot_trial(corridors, references, trial, output):
    genome = np.array([trial["champion"]])
    paths = {}
    for name, geometry in corridors.items():
        _, finished, progress, path = drive(genome, True, geometry, trace=True,
                                           steering_scale=90 / 35, input_scale=pilot.SENSOR_RANGE / SCALE)
        saved = trial["courses"][name]
        if (bool(finished[0]) != saved["finished"] or not np.isclose(progress[0], saved["progress"])
                or len(path) - 1 != saved["ticks"]):
            raise ValueError(f"Could not reproduce {name} from the archived trial")
        paths[name] = (bool(finished[0]), np.asarray(path))
    plot_geometry(corridors, references, output, trial_paths=paths,
                  development=trial.get("evaluation_role") == "development")


if __name__ == "__main__":
    main()
