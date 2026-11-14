"""AE4350 pilot: evolve free and mirror-constrained steering."""

import argparse
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ae4350-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


START = np.array([5.0, 7.0])
WHEELBASE = 2.6
HALF_LENGTH, HALF_WIDTH = 2.15, 0.85
STEP_DISTANCE = 4.0 * 0.1  # 4 m/s, 0.1 s
STEERING_LIMIT = np.deg2rad(35)
MAX_TICKS = 220
SENSOR_RANGE = 25.0
BEAM_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 25)
HIDDEN = 15
POPULATION = 200
COURSE_WIDTH = 14.0
LOW_RESOLUTION_INDICES = np.arange(0, 25, 4)
LOW_RESOLUTION_INTERPOLATION = np.array([
    np.interp(np.arange(25), LOW_RESOLUTION_INDICES, basis)
    for basis in np.eye(len(LOW_RESOLUTION_INDICES))
])


def axis_segments(boxes, axis):
    """Face coordinates and transverse bounds, with duplicate/empty faces removed."""
    transverse = 1 - axis
    lower = boxes[:, transverse]
    upper = boxes[:, transverse + 2]
    near = np.column_stack([boxes[:, axis], lower, upper])
    far = np.column_stack([boxes[:, axis + 2], lower, upper])
    faces = np.unique(np.concatenate([near, far]), axis=0)
    return faces[faces[:, 1] < faces[:, 2]].T


@dataclass
class Course:
    length: float
    finish_x: float
    obstacles: np.ndarray
    centers: np.ndarray = field(init=False)
    halves: np.ndarray = field(init=False)
    vertical: np.ndarray = field(init=False)
    horizontal: np.ndarray = field(init=False)

    def __post_init__(self):
        length, width = self.length, COURSE_WIDTH
        walls = [[0, 0, 0, width], [length, 0, length, width],
                 [0, 0, length, 0], [0, width, length, width]]
        boxes = np.concatenate([self.obstacles, walls])
        self.centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        self.halves = (boxes[:, 2:] - boxes[:, :2]) / 2
        self.vertical = axis_segments(boxes, 0)
        self.horizontal = axis_segments(boxes, 1)


DEFAULT_COURSE = Course(68.0, 64.0, np.array([[22, 0, 25, 7.5], [45, 6.5, 48, 14]]))
TRAINING_COURSES = (
    Course(68.0, 64.0, np.array([[22, 0, 25, 7.5], [41, 6.5, 44, 14]])),
    Course(68.0, 64.0, np.array([[20, 6.5, 23, 14], [43, 0, 46, 7.5]])),
)
TRAINING_SCENARIOS = tuple(
    (course, np.array([5.0, start_y]))
    for course in TRAINING_COURSES
    for start_y in (6.5, 7.5)
)


def direction(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def face_ranges(origin, rays, axis, faces):
    transverse = 1 - axis
    location, low, high = faces
    component = rays[:, :, axis, None]
    parallel = np.abs(component) < 1e-12
    distance = (location - origin[:, None, axis, None]) / np.where(parallel, 1, component)
    crossing = origin[:, None, transverse, None] + distance * rays[:, :, transverse, None]
    hit = ~parallel & (distance >= 0) & (crossing >= low) & (crossing <= high)
    return np.min(np.where(hit, distance, SENSOR_RANGE), axis=-1)


def sense(position, heading, course=DEFAULT_COURSE, *, physical_beams=25, relative_noise=None):
    origin = position + HALF_LENGTH * direction(heading)
    indices = LOW_RESOLUTION_INDICES if physical_beams == 7 else slice(None)
    rays = direction(heading[:, None] + BEAM_ANGLES[indices])
    vertical = face_ranges(origin, rays, 0, course.vertical)
    horizontal = face_ranges(origin, rays, 1, course.horizontal)
    readings = np.minimum(SENSOR_RANGE, np.minimum(vertical, horizontal)) / SENSOR_RANGE
    if relative_noise is not None:
        readings = np.clip(readings * (1 + relative_noise[indices]), 0, 1)
    if physical_beams == 7:
        return readings @ LOW_RESOLUTION_INTERPOLATION
    return readings


def move(position, heading, steering):
    forward = direction(heading)
    rear = position - (WHEELBASE / 2) * forward + STEP_DISTANCE * forward
    front = position + (WHEELBASE / 2) * forward + STEP_DISTANCE * direction(heading + steering)
    axle = front - rear
    return (front + rear) / 2, np.arctan2(axle[:, 1], axle[:, 0])


def collides(position, heading, course=DEFAULT_COURSE):
    """Rectangle overlap by projection on the world and vehicle axes."""
    offset = course.centers - position[:, None, :]
    forward = direction(heading)
    lateral = direction(heading + np.pi / 2)
    body_world_half = HALF_LENGTH * np.abs(forward) + HALF_WIDTH * np.abs(lateral)
    world_overlap = np.all(np.abs(offset) <= course.halves + body_world_half[:, None, :], axis=-1)
    forward_overlap = np.abs(np.einsum("nbi,ni->nb", offset, forward)) <= (
        HALF_LENGTH + np.abs(forward) @ course.halves.T
    )
    lateral_overlap = np.abs(np.einsum("nbi,ni->nb", offset, lateral)) <= (
        HALF_WIDTH + np.abs(lateral) @ course.halves.T
    )
    return np.any(world_overlap & forward_overlap & lateral_overlap, axis=1)


def decode(genomes, symmetric, hidden=HIDDEN):
    input_count = 12 if symmetric else 25
    split = input_count * hidden
    inputs = genomes[:, :split].reshape(-1, hidden, input_count)
    outputs = genomes[:, split:].reshape(-1, 2, hidden)
    return inputs, outputs


def command(scan, inputs, outputs, symmetric):
    if symmetric:
        scan = scan[:, :12] - scan[:, 13:][:, ::-1]
    hidden = 0.5 * np.tanh(np.einsum("ni,nhi->nh", scan, inputs) / 2)
    steering_outputs = 0.5 * np.tanh(np.einsum("nh,noh->no", hidden, outputs) / 2)
    return STEERING_LIMIT * (steering_outputs[:, 1] - steering_outputs[:, 0])


def fitness(progress, ticks):
    return progress**2 / np.maximum(1, ticks)


def evaluate(genomes, symmetric, record=False, *, course=DEFAULT_COURSE, start=START,
             hidden=HIDDEN, physical_beams=25, noise=None):
    count = len(genomes)
    inputs, outputs = decode(genomes, symmetric, hidden)
    position = np.tile(start, (count, 1))
    heading = np.zeros(count)
    ticks = np.zeros(count, dtype=int)
    finished = np.zeros(count, dtype=bool)
    active = np.arange(count)
    path = [start.copy()] if record else None

    for tick in range(MAX_TICKS):
        scan = sense(position[active], heading[active], course, physical_beams=physical_beams,
                     relative_noise=None if noise is None else noise[tick])
        steering = command(scan, inputs[active], outputs[active], symmetric)
        next_position, next_heading = move(position[active], heading[active], steering)
        collision = collides(next_position, next_heading, course)
        safe = active[~collision]
        position[safe] = next_position[~collision]
        heading[safe] = next_heading[~collision]
        ticks[active] += 1
        finished[safe] = position[safe, 0] >= course.finish_x
        if record:
            path.append(next_position[0].copy())
        active = safe[~finished[safe]]
        if not len(active):
            break

    progress = np.clip(position[:, 0] - start[0], 0, course.finish_x - start[0])
    return fitness(progress, ticks), finished, path


@dataclass(frozen=True)
class GASettings:
    population: int = POPULATION
    tournament_size: int = 10
    mutation_probability_per_gene: float = 0.10
    weight_min: float = -1.0
    weight_max: float = 1.0
    crossover_fraction_mean: float = 0.95
    crossover_fraction_std: float = 0.05


def offspring(genomes, scores, rng, ga=GASettings(), elites=0):
    population, genes = genomes.shape
    tournament = rng.integers(population, size=(population, ga.tournament_size))
    winners = tournament[np.arange(population), np.argmax(scores[tournament], axis=1)]
    first, second = rng.choice(winners, size=(2, (population - elites) // 2))
    fraction = rng.normal(ga.crossover_fraction_mean, ga.crossover_fraction_std, len(first))
    cut = np.clip(np.rint(fraction * genes), 1, genes - 1)
    prefix = np.arange(genes) < cut[:, None]
    children = np.concatenate([
        np.where(prefix, genomes[first], genomes[second]),
        np.where(prefix, genomes[second], genomes[first]),
    ])
    mutate = rng.random(children.shape) < ga.mutation_probability_per_gene
    children[mutate] = rng.uniform(ga.weight_min, ga.weight_max, np.count_nonzero(mutate))
    if elites:
        best = np.argsort(-scores, kind="stable")[:elites]
        return np.concatenate([genomes[best], children])
    return children


def evaluate_scenarios(genomes, symmetric, scenarios, *, hidden=HIDDEN, physical_beams=25):
    scores = []
    finishes = []
    for course, start in scenarios:
        score, finished, _ = evaluate(genomes, symmetric, course=course, start=start,
                                      hidden=hidden, physical_beams=physical_beams)
        scores.append(score)
        finishes.append(finished)
    return np.mean(scores, axis=0), np.stack(finishes, axis=1)


@dataclass
class SearchResult:
    history: np.ndarray
    champion: np.ndarray
    champion_finishes: np.ndarray
    first_full_success: int | None
    initial_full_successes: int


def search(symmetric, generations, seed, scenarios, *, hidden=HIDDEN, physical_beams=25,
           ga=GASettings(), elites=0):
    rng = np.random.default_rng(seed)
    gene_count = hidden * (12 if symmetric else 25) + 2 * hidden
    genomes = rng.uniform(ga.weight_min, ga.weight_max, (ga.population, gene_count))
    history = []
    best_score = -np.inf
    first_full_success = None
    for population_number in range(1, generations + 1):
        scores, finished = evaluate_scenarios(genomes, symmetric, scenarios,
                                              hidden=hidden, physical_beams=physical_beams)
        full_successes = np.count_nonzero(finished.all(axis=1))
        if population_number == 1:
            initial_full_successes = int(full_successes)
        winner = np.argmax(scores)
        if scores[winner] > best_score:
            best_score = scores[winner]
            champion = genomes[winner:winner + 1].copy()
            champion_finishes = finished[winner].copy()
        history.append(best_score)
        if first_full_success is None and full_successes:
            first_full_success = population_number
        if population_number < generations:
            genomes = offspring(genomes, scores, rng, ga, elites)
    return SearchResult(np.array(history), champion, champion_finishes, first_full_success,
                        initial_full_successes)


def evolve(symmetric, generations, seed):
    run = search(symmetric, generations, seed, ((DEFAULT_COURSE, START),))
    _, finished, path = evaluate(run.champion, symmetric, record=True)
    return run.history, np.array(path), bool(finished[0]), run.first_full_success is not None


def check_pilot():
    rng = np.random.default_rng(4350)
    inputs, outputs = decode(rng.uniform(-1, 1, (20, 210)), symmetric=True)
    scans = rng.random((20, 25))
    nominal = command(scans, inputs, outputs, True)
    np.testing.assert_allclose(command(scans[:, ::-1], inputs, outputs, True), -nominal)
    scans[:, 12] = 1 - scans[:, 12]
    np.testing.assert_allclose(command(scans, inputs, outputs, True), nominal)
    start = START[None, :]
    np.testing.assert_allclose(sense(start, np.zeros(1))[0, 12], (22 - 5 - HALF_LENGTH) / SENSOR_RANGE)
    position, heading = move(np.repeat(start, 3, axis=0), np.zeros(3), np.array([0, 0.3, -0.3]))
    np.testing.assert_allclose(position[0], START + [STEP_DISTANCE, 0])
    assert heading[1] > 0 > heading[2]
    np.testing.assert_allclose(heading[1], -heading[2])
    assert not collides(start, np.zeros(1))[0]
    assert collides(np.array([[START[0], HALF_WIDTH], [22 - HALF_LENGTH, 7]]), np.zeros(2)).all()
    finish_score = fitness(DEFAULT_COURSE.finish_x - START[0], 160)
    assert finish_score > fitness(14, 35) > fitness(0, MAX_TICKS)
    straight_score, finished, _ = evaluate(np.zeros((1, 405)), symmetric=False)
    assert not finished[0] and 0 < straight_score[0] < finish_score
    print("Checks passed: mirror, center beam, range, motion, contact, and fitness.", flush=True)


def save_plot(results, seeds, output):
    fig, (learning, course) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = ["#0076a8", "#b84614"]
    for (label, runs), color, linestyle in zip(results.items(), colors, ["-", "--"]):
        histories = np.stack([run[0] for run in runs])
        evaluations = POPULATION * np.arange(1, histories.shape[1] + 1)
        learning.plot(evaluations, np.median(histories, axis=0), label=label,
                      color=color, linestyle=linestyle)
        learning.fill_between(evaluations, histories.min(axis=0), histories.max(axis=0), color=color, alpha=0.16)
        _, path, finished, _ = runs[0]
        outcome = "finished" if finished else "did not finish"
        course.plot(path[:, 0], path[:, 1], color=color, linestyle=linestyle,
                    label=f"{label} · {outcome}")
        course.scatter(*path[-1], color=color, s=22, zorder=4)

    for x0, y0, x1, y1 in DEFAULT_COURSE.obstacles:
        course.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#374151"))
    course.axvline(DEFAULT_COURSE.finish_x, color="#167144", linestyle="--", linewidth=1, label="Finish")
    course.scatter(*START, color="#111827", s=22, zorder=4, label="Start")
    course.set(xlim=(0, DEFAULT_COURSE.length), ylim=(0, COURSE_WIDTH), aspect="equal", xlabel="x [m]", ylabel="y [m]",
               title=f"Best-fitness trajectories · seed {seeds[0]}")
    course.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2, fontsize=8)
    learning.set(xlabel="Candidate evaluations", ylabel="Best fitness so far [m² / tick]",
                 title=f"Learning · median and range of {len(seeds)} runs")
    learning.grid(alpha=0.2)
    learning.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true", help="Train on two courses with two starts each")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4350, 4351, 4352])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pilot_results.png"))
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be at least 1")
    check_pilot()
    if args.suite:
        for symmetric, label in [(False, "Unconstrained"), (True, "Symmetric")]:
            for seed in args.seeds:
                run = search(symmetric, args.generations, seed, TRAINING_SCENARIOS)
                finishes = run.champion_finishes
                print(f"{label}, seed={seed}: mean fitness {run.history[0]:.2f} -> {run.history[-1]:.2f}, "
                      f"champion finishes={finishes.sum()}/{len(finishes)}, "
                      f"first full success={run.first_full_success}", flush=True)
        return

    results = {}
    for symmetric, label in [(False, "Unconstrained"), (True, "Symmetric")]:
        runs = []
        for seed in args.seeds:
            run = evolve(symmetric, args.generations, seed)
            history, _, champion_finished, any_finished = run
            runs.append(run)
            print(f"{label}, seed={seed}: fitness {history[0]:.2f} -> {history[-1]:.2f}, "
                  f"champion finished={champion_finished}, any finished={any_finished}", flush=True)
        results[label] = runs
    save_plot(results, args.seeds, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
