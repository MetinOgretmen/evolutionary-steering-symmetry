"""Evolve one reactive controller on the four reconstructed development courses."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from numba import njit, prange

import pilot
from paper_corridors import ROOT, SCALE, VehicleSetup, drive, load_corridors, plot_trial, reference
from run_experiments import save_json


RANGE = pilot.SENSOR_RANGE
BEAM_ANGLES = pilot.BEAM_ANGLES
HALF_LENGTH = pilot.HALF_LENGTH
HALF_WIDTH = pilot.HALF_WIDTH
STEP = pilot.STEP_DISTANCE
WHEELBASE = pilot.WHEELBASE
STEERING_SCALE = math.pi / 2


@njit
def range_faces(faces, origin, transverse, component, other_component, best):
    """Mirror the Python ray–segment intersection on one wall orientation."""
    if abs(component) > 1e-12:
        for i in range(faces.shape[1]):
            distance = (faces[0, i] - origin) / component
            if 0 <= distance < best:
                crossing = transverse + distance * other_component
                if faces[1, i] <= crossing <= faces[2, i]:
                    best = distance
    return best


@njit
def dense_command(genome, readings, input_count, hidden, reflected=False):
    output_start = input_count * hidden
    left = 0.0
    right = 0.0
    for h in range(hidden):
        total = 0.0
        for j in range(input_count):
            index = input_count - j - 1 if reflected else j
            total += readings[index] * genome[h * input_count + j]
        value = 0.5 * math.tanh(total / 2)
        left += value * genome[output_start + h]
        right += value * genome[output_start + hidden + h]
    return STEERING_SCALE * 0.5 * (math.tanh(right / 2) - math.tanh(left / 2))


@njit
def steering(genome, vertical, horizontal, x, y, angle, cosine, sine, sensor_offset=HALF_LENGTH,
             symmetric=True, hidden=15, physical_beams=25, relative_noise=None, projected=False):
    origin_x = x + sensor_offset * cosine
    origin_y = y + sensor_offset * sine
    readings = np.empty(25)
    stride = 4 if physical_beams == 7 else 1
    for j in range(0, 25, stride):
        if j == 12 and symmetric and stride == 1:
            continue
        direction = angle + BEAM_ANGLES[j]
        dx, dy = math.cos(direction), math.sin(direction)
        distance = range_faces(vertical, origin_x, origin_y, dx, dy, RANGE)
        distance = range_faces(horizontal, origin_y, origin_x, dy, dx, distance)
        reading = distance / RANGE
        if relative_noise is not None:
            reading = min(1.0, max(0.0, reading * (1 + relative_noise[j])))
        readings[j] = reading * (RANGE / SCALE)
    if stride == 4:
        for base in range(0, 24, 4):
            for offset in range(1, 4):
                fraction = offset / 4
                readings[base + offset] = (1 - fraction) * readings[base] + fraction * readings[base + 4]
    if symmetric:
        for j in range(12):
            readings[j] -= readings[24 - j]
    input_count = 12 if symmetric else 25
    command = dense_command(genome, readings, input_count, hidden)
    if projected:
        command = (command - dense_command(genome, readings, input_count, hidden, True)) / 2
    return command


@njit
def contact(centers, halves, nearby, x, y, cosine, sine, padding=0.0):
    ac, ass = abs(cosine), abs(sine)
    length, width = HALF_LENGTH + padding, HALF_WIDTH + padding
    world_x = length * ac + width * ass
    world_y = length * ass + width * ac
    for index in nearby:
        if index < 0:
            break
        ox, oy = centers[index, 0] - x, centers[index, 1] - y
        hx, hy = halves[index, 0], halves[index, 1]
        if abs(ox) > hx + world_x or abs(oy) > hy + world_y:
            continue
        if (abs(ox * cosine + oy * sine) <= length + hx * ac + hy * ass
                and abs(-ox * sine + oy * cosine) <= width + hx * ass + hy * ac):
            return True
    return False


@njit
def pose_contact(centers, halves, cells, x, y, cosine, sine, body_offset, padding):
    body_x = x + body_offset * cosine
    body_y = y + body_offset * sine
    row, column = int(-body_y / 4), int(body_x / 4)
    if row < 0 or column < 0 or row >= cells.shape[0] or column >= cells.shape[1]:
        return True
    return contact(centers, halves, cells[row, column], body_x, body_y, cosine, sine, padding)


@njit(parallel=True)
def simulate(genomes, vertical, horizontal, centers, halves, cells, distance,
             start, goal, heading, length, max_ticks, step=STEP,
             sensor_offset=HALF_LENGTH, body_offset=0.0, padding=0.0,
             symmetric=True, hidden=15, physical_beams=25, noise=None, targets=None, pose_samples=1,
             projected=False, adaptive_speed=False, clearance_cost=False, exit_gate=None,
             exit_remaining=0.0):
    progress = np.empty(len(genomes))
    finished = np.zeros(len(genomes), dtype=np.bool_)
    ticks = np.empty(len(genomes), dtype=np.int64)
    steering_error = np.zeros(len(genomes))
    close_poses = np.zeros(len(genomes))
    for i in prange(len(genomes)):
        x, y = start
        angle = heading
        for tick in range(1, max_ticks + 1):
            cosine, sine = math.cos(angle), math.sin(angle)
            delta = steering(genomes[i], vertical, horizontal, x, y, angle, cosine, sine, sensor_offset,
                             symmetric, hidden, physical_beams,
                             None if noise is None else noise[tick - 1], projected)
            distance_step = step
            if adaptive_speed:
                ox, oy = x + sensor_offset * cosine, y + sensor_offset * sine
                front_range = range_faces(vertical, ox, oy, cosine, sine, RANGE)
                front_range = range_faces(horizontal, oy, ox, sine, cosine, front_range)
                if noise is not None:
                    front_range = min(RANGE, max(0.0, front_range * (1 + noise[tick - 1, 12])))
                factor = min(front_range / (2 * WHEELBASE), 1 - abs(delta) / (math.pi / 3))
                distance_step = step * min(1.0, max(0.4, factor))
            if targets is not None:
                target = targets[round(-y / SCALE), round(x / SCALE)]
                dx, dy = target[0] - x, target[1] - y
                reference = math.atan2(2 * WHEELBASE * (cosine * dy - sine * dx), dx**2 + dy**2)
                steering_error[i] += abs(delta - reference)
            front_cos, front_sin = math.cos(angle + delta), math.sin(angle + delta)
            rear_x = x - (WHEELBASE / 2) * cosine + distance_step * cosine
            rear_y = y - (WHEELBASE / 2) * sine + distance_step * sine
            front_x = x + (WHEELBASE / 2) * cosine + distance_step * front_cos
            front_y = y + (WHEELBASE / 2) * sine + distance_step * front_sin
            next_x = (front_x + rear_x) / 2
            next_y = (front_y + rear_y) / 2
            next_angle = math.atan2(front_y - rear_y, front_x - rear_x)
            next_cos, next_sin = math.cos(next_angle), math.sin(next_angle)
            if pose_contact(centers, halves, cells, next_x, next_y, next_cos, next_sin, body_offset, padding):
                break
            if pose_samples > 1:
                heading_change = (next_angle - angle + math.pi) % (2 * math.pi) - math.pi
                blocked = False
                for sample in range(1, pose_samples):
                    fraction = sample / pose_samples
                    intermediate = angle + fraction * heading_change
                    if pose_contact(centers, halves, cells, x + fraction * (next_x - x),
                                    y + fraction * (next_y - y), math.cos(intermediate),
                                    math.sin(intermediate), body_offset, padding):
                        blocked = True
                        break
                if blocked:
                    break
            if clearance_cost:
                # Penalize grazing without declaring a non-contact pose a collision.
                if pose_contact(centers, halves, cells, next_x, next_y, next_cos, next_sin,
                                body_offset, padding + SCALE / 2):
                    close_poses[i] += 1
                elif pose_contact(centers, halves, cells, next_x, next_y, next_cos, next_sin,
                                  body_offset, padding + SCALE):
                    close_poses[i] += 0.5
            x, y, angle = next_x, next_y, next_angle
            if exit_gate is None:
                reached_exit = (x - goal[0])**2 + (y - goal[1])**2 <= 9
            else:
                gx, gy, fx, fy, half_width = exit_gate
                body_dx = x + body_offset * next_cos - gx
                body_dy = y + body_offset * next_sin - gy
                along = next_cos * fx + next_sin * fy
                across = next_cos * fy - next_sin * fx
                rear = body_dx * fx + body_dy * fy - HALF_LENGTH * abs(along) - HALF_WIDTH * abs(across)
                reached_exit = rear >= 0 and abs(body_dx * fy - body_dy * fx) <= half_width
            if reached_exit:
                finished[i] = True
                break
        ticks[i] = tick
        remaining = distance[round(-y / SCALE), round(x / SCALE)]
        progress[i] = 1 if finished[i] else min(1.0, max(0.0, 1 - (remaining - exit_remaining) / length))
    return progress, finished, ticks, steering_error / ticks, close_poses / ticks


@dataclass
class PreparedCourse:
    course: object

    def __post_init__(self):
        geometry = self.course
        boxes = geometry.boxes
        columns = int(geometry.mask.shape[1] * SCALE / 4) + 1
        rows = int(geometry.mask.shape[0] * SCALE / 4) + 1
        nearby = []
        margin = math.hypot(HALF_LENGTH + 0.5, HALF_WIDTH + 0.5)
        for row in range(rows):
            for column in range(columns):
                x, y = 4 * column, -4 * row
                nearby.append(np.flatnonzero(
                    (boxes[:, 0] <= x + 4 + margin) & (boxes[:, 2] >= x - margin)
                    & (boxes[:, 1] <= y + margin) & (boxes[:, 3] >= y - 4 - margin)))
        self.cells = np.full((rows * columns, max(map(len, nearby))), -1, dtype=np.int64)
        for index, candidates in enumerate(nearby):
            self.cells[index, :len(candidates)] = candidates
        self.cells = self.cells.reshape(rows, columns, -1)

    def evaluate(self, genomes, vehicle=VehicleSetup(), padding=0.0, *,
                 symmetric=True, hidden=15, physical_beams=25, noise=None, scenario=None, targets=None,
                 pose_samples=1, projected=False, adaptive_speed=False, clearance_cost=False,
                 exit_gate=None, exit_remaining=0.0):
        if clearance_cost and padding:
            raise ValueError("Clearance scoring uses the unpadded body for actual collisions.")
        course = self.course if scenario is None else scenario
        result = simulate(genomes, course.vertical, course.horizontal, course.centers, course.halves, self.cells,
                          course.distance_to_goal, course.start, course.goal, course.heading,
                          course.length, course.max_ticks, vehicle.step_distance, vehicle.sensor_offset,
                          vehicle.body_offset, padding, symmetric, hidden, physical_beams, noise, targets,
                          pose_samples, projected, adaptive_speed, clearance_cost, exit_gate, exit_remaining)
        return result[:3] if targets is None and not clearance_cost else result


def archive_genomes(results):
    seeds = (4350, 5350, 5351, 5352, 5353)
    archived = []
    for seed in seeds:
        path = results / f"paper_trial_source_inspired_symmetric_{seed}.json"
        archived.append(json.loads(path.read_text())["champion"])
    normalized = json.loads((results / "paper_trial_pilot_symmetric_4350.json").read_text())["champion"]
    normalized = np.array(normalized)
    normalized[:180] *= SCALE / RANGE
    # Preserve the old output range for small activations as a warm start.
    normalized[180:] *= 35 / 90
    return np.array([*archived, normalized])


def verified_trial(genome, courses):
    return {
        name: {"finished": bool(finished[0]), "progress": float(progress[0]), "ticks": len(path) - 1}
        for name, course in courses.items()
        for _, finished, progress, path in [
            drive(genome[None, :], True, course, steering_scale=90 / 35, input_scale=RANGE / SCALE, trace=True)
        ]
    }


def save_trial(result, courses, path, figure=False):
    save_json(path, result)
    if figure:
        references = {}
        for name, course in courses.items():
            success, route = reference(course, 5.0)
            if not success:
                raise ValueError(f"Could not reconstruct reference path for {name}")
            references[name] = route
        plot_trial(courses, references, result, path.with_suffix(".png"))


def search(courses, archive, seed, generations, population, output):
    rng = np.random.default_rng(seed)
    prepared = [PreparedCourse(course) for course in courses.values()]
    # The compiled evaluator proposes genomes. The original Python evaluator
    # decides which ones can be reported, including numerical edge cases.
    disagreements = 0
    for i, genome in enumerate(archive[:5]):
        reference_rows = verified_trial(genome, courses)
        fast = [course.evaluate(genome[None, :]) for course in prepared]
        for (name, expected), (progress, finish, ticks) in zip(reference_rows.items(), fast):
            if bool(finish[0]) != expected["finished"] or abs(progress[0] - expected["progress"]) > 0.002:
                disagreements += 1
                print(f"Fast and reference disagree on archived genome {i}, {name}.", flush=True)
    print(f"Search preflight: {20 - disagreements}/20 within tolerance; finalists use Python replay.",
          flush=True)
    genomes = archive.copy()
    best_key = (-1, -1.0, -1.0)
    checked = set()
    for generation in range(generations + 1):
        scores = [course.evaluate(genomes) for course in prepared]
        progress = np.stack([row[0] for row in scores], axis=1)
        finishes = np.stack([row[1] for row in scores], axis=1)
        counts = finishes.sum(axis=1)
        mean = progress.mean(axis=1)
        minimum = progress.min(axis=1)
        order = np.lexsort((-mean, -minimum, -counts))
        for best in order[:6]:
            proposal = (int(counts[best]), float(minimum[best]), float(mean[best]))
            signature = genomes[best].tobytes()
            if signature in checked or (proposal <= best_key and generation % 10 != 0):
                continue
            checked.add(signature)
            verified = verified_trial(genomes[best], courses)
            distances = np.array([row["progress"] for row in verified.values()])
            key = (sum(row["finished"] for row in verified.values()), distances.min(), distances.mean())
            if key <= best_key:
                continue
            best_key = key
            print(f"Generation {generation}: verified {key[0]}/4, "
                  f"progress {np.round(100 * distances, 1).tolist()}", flush=True)
            trial = {
                "generation": generation,
                "search_seed": seed,
                "population": population,
                "champion": genomes[best].tolist(),
                "courses": verified,
                "evaluation_role": "development",
            }
            save_trial(trial, courses, output)
            if all(row["finished"] for row in verified.values()):
                save_trial(trial, courses, output, figure=True)
                return
        if generation % 10 == 0:
            print(f"Generation {generation}/{generations} complete.", flush=True)
        elite_indices = order[:12]
        # Keep specialists for each course as well as the best joint performers.
        specialists = np.concatenate([np.argsort(-progress[:, i], kind="stable")[:3]
                                      for i in range(progress.shape[1])])
        parents = genomes[np.unique(np.r_[elite_indices, specialists])]
        count = population - len(parents)
        children = parents[rng.integers(len(parents), size=count)].copy()
        step_sizes = rng.choice([0.005, 0.02, 0.07, 0.20], size=(count, 1))
        mutation = rng.random(children.shape) < 0.22
        children += mutation * step_sizes * rng.standard_normal(children.shape)
        genomes = np.concatenate([parents, children])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--seed", type=int, default=6250)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "corridor_development.json")
    args = parser.parse_args()
    if args.population < 24 or args.generations < 0:
        parser.error("At least 24 candidates and zero or more generations are required.")
    search(load_corridors(), archive_genomes(ROOT / "results"), args.seed,
           args.generations, args.population, args.output)


if __name__ == "__main__":
    main()
