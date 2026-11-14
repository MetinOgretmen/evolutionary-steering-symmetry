"""Replay frozen controllers with local LiDAR and a short chassis safety check."""

from collections import deque
import argparse
import json

import numpy as np

import pilot
from paper_corridors import ROOT, SCALE, load_corridors, move_vehicle
from run_experiments import save_json
from source_reproduction import OPTIONS
from transfer_experiment import SEEDS


VEHICLE = OPTIONS["vehicle"]
HORIZON = 3
PHASES = np.arange(1, 11) / 10
SAFETY_MARGIN = 0.12
STEERING_ALTERNATIVES = np.deg2rad([0, -10, 10, -20, 20, -35, 35, -50, 50, -60, 60])


def scan_points(position, heading, ranges):
    distances = ranges * pilot.SENSOR_RANGE
    endpoints = position + distances[:, None] * pilot.direction(heading + pilot.BEAM_ANGLES)
    hit = distances < pilot.SENSOR_RANGE - 1e-9
    left, right = endpoints[:-1], endpoints[1:]
    joins = hit[:-1] & hit[1:] & (np.linalg.norm(right - left, axis=1) <= 1.5)
    phases = np.arange(1, 10) / 10
    interiors = left[joins, None, :] + phases[None, :, None] * (right[joins] - left[joins])[:, None, :]
    return np.concatenate((endpoints[hit], interiors.reshape(-1, 2)))


def between(position, heading, end, end_heading, phases):
    rotation = (end_heading - heading + np.pi) % (2 * np.pi) - np.pi
    xy = position[:, None] + phases[None, :, None] * (end - position)[:, None]
    angles = heading[:, None] + phases[None, :] * rotation[:, None]
    bodies = xy + VEHICLE.body_offset * pilot.direction(angles)
    return bodies, angles


def safe_horizon(position, heading, angles, points, step_distance=VEHICLE.step_distance, *,
                 horizon=HORIZON, hold_turn=False, margin=SAFETY_MARGIN):
    positions = np.repeat(position, len(angles), axis=0)
    headings = np.repeat(heading, len(angles))
    clear_steps = np.full(len(angles), horizon)
    for step in range(horizon):
        turn = angles if step == 0 or hold_turn else np.zeros_like(angles)
        end, end_heading = move_vehicle(positions, headings, turn, step_distance)
        body, orientation = between(positions, headings, end, end_heading, PHASES[1::2])
        forward = pilot.direction(orientation)
        side = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
        relative = points[None, None] - body[:, :, None]
        x = np.einsum("nspi,nsi->nsp", relative, forward)
        y = np.einsum("nspi,nsi->nsp", relative, side)
        touches = np.any((np.abs(x) <= pilot.HALF_LENGTH + margin)
                         & (np.abs(y) <= pilot.HALF_WIDTH + margin), axis=(1, 2))
        clear_steps[touches & (clear_steps == horizon)] = step
        positions, headings = end, end_heading
    return clear_steps


def local_points(position, history):
    points = np.concatenate(tuple(history))
    points = points[np.sum((points - position[0])**2, axis=1) < 7**2]
    # Quantize and deduplicate repeated returns to keep the local estimate small.
    return np.unique(np.rint(points / 0.10), axis=0) * 0.10


def guard_command(position, heading, proposed, history, *, recover=False):
    points = local_points(position, history)
    step_distance = VEHICLE.step_distance
    if safe_horizon(position, heading, np.array([proposed]), points)[0] == HORIZON:
        return proposed, step_distance
    choices = np.r_[proposed, STEERING_ALTERNATIVES]
    horizon = safe_horizon(position, heading, choices, points)
    # Keep the learned action whenever no alternative buys a longer safe horizon.
    order = np.lexsort((np.abs(choices - proposed), -horizon))
    winner = order[0]
    if horizon[winner] == HORIZON or (not recover and horizon[winner] > 0):
        return float(choices[winner]), step_distance
    if recover:
        short = pilot.STEP_DISTANCE
        count = int(np.ceil(HORIZON * step_distance / short))
        straight = safe_horizon(position, heading, choices, points, short, horizon=count)
        turning = safe_horizon(position, heading, choices, points, short, horizon=count, hold_turn=True)
        short_distance = np.minimum(HORIZON * step_distance, np.maximum(straight, turning) * short)
        short_winner = np.lexsort((np.abs(choices - proposed), -short_distance))[0]
        if short_distance[short_winner] > horizon[winner] * step_distance:
            return float(choices[short_winner]), short
        if horizon[winner] > 0:
            return float(choices[winner]), step_distance
    return None, 0.0


def terminal_command(position, heading, scan, goal):
    """Aim for a supplied goal only when it is close and visible in the front scan."""
    offset = goal - position
    distance = np.linalg.norm(offset)
    if distance > 4 * pilot.HALF_LENGTH:
        return None
    angle = (np.arctan2(offset[1], offset[0]) - heading + np.pi) % (2 * np.pi) - np.pi
    if not pilot.BEAM_ANGLES[0] < angle < pilot.BEAM_ANGLES[-1]:
        return None
    right = np.searchsorted(pilot.BEAM_ANGLES, angle)
    if min(scan[right - 1], scan[right]) * pilot.SENSOR_RANGE < distance:
        return None
    return float(np.arctan2(2 * pilot.WHEELBASE * np.sin(angle), distance))


def rollout(genome, course, *, guarded=False, straight_ticks=(), symmetric=True, recover=False, goal_assist=False):
    inputs, outputs = pilot.decode(genome, symmetric)
    position = course.start[None].copy()
    heading = np.asarray([course.heading])
    history = deque(maxlen=7)
    path, commands = [position[0].tolist()], []
    interventions = 0
    short_steps = 0
    goal_steps = 0
    for tick in range(course.max_ticks):
        scan = pilot.sense(position - pilot.HALF_LENGTH * pilot.direction(heading), heading, course)
        proposed = float((OPTIONS["steering_scale"] * pilot.command(scan * OPTIONS["input_scale"],
                                                                   inputs, outputs, symmetric))[0])
        command = proposed
        step_distance = VEHICLE.step_distance
        if goal_assist:
            target_turn = terminal_command(position[0], heading[0], scan[0], course.goal)
            if target_turn is not None:
                command = target_turn
                goal_steps += 1
        if guarded:
            history.append(scan_points(position[0], heading[0], scan[0]))
            command, step_distance = guard_command(position, heading, command, history, recover=recover)
            interventions += command != proposed or step_distance != VEHICLE.step_distance
        if tick in straight_ticks:
            command = 0.0
        if command is None:
            reason = "stopped_by_guard"
            break
        short_steps += step_distance < VEHICLE.step_distance
        end, end_heading = move_vehicle(position, heading, np.array([command]), step_distance)
        bodies, angles = between(position, heading, end, end_heading, PHASES)
        contact = pilot.collides(bodies[0], angles[0], course)
        commands.append([float(np.rad2deg(proposed)), float(np.rad2deg(command)), bool(contact.any())])
        path.append(end[0].tolist())
        if contact.any():
            reason = "body_contact"
            break
        position, heading = end, end_heading
        if np.linalg.norm(position[0] - course.goal) <= 3.0:
            reason = "finished"
            break
    else:
        reason = "timeout"
    x, y = np.rint(position[0] / [SCALE, -SCALE]).astype(int)
    progress = np.clip(1 - course.distance_to_goal[y, x] / course.length, 0, 1)
    return {"reason": reason, "ticks": len(commands), "progress": 1.0 if reason == "finished" else float(progress),
            "interventions": interventions, "short_steps": short_steps, "goal_steps": goal_steps,
            "path": path, "commands": commands}


def weights(seed, architecture="symmetric", experiment="transfer"):
    path = ROOT / "results" / experiment / f"weights_{architecture}_restarts_{seed}.json"
    return np.asarray([json.loads(path.read_text())["champion"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", action="store_true", help="Replay the matched unconstrained networks")
    parser.add_argument("--verified-selection", action="store_true",
                        help="Replay the networks selected with exact training verification")
    args = parser.parse_args()
    courses = load_corridors()
    training = courses.pop("map6")
    directory = ROOT / "results" / "transfer_guard"
    directory.mkdir(exist_ok=True)
    # The straight intervention is diagnostic and uses the known contact time.
    if not args.free and not args.verified_selection:
        example = rollout(weights(7311), courses["map9"])
        assert example["reason"] == "body_contact"
        contact = example["ticks"] - 1
        straight = rollout(weights(7311), courses["map9"], straight_ticks=range(contact, contact + 2))
        guarded = rollout(weights(7311), courses["map9"], guarded=True)
        save_json(directory / "corner_example.json", {"original": example, "straight_for_two_steps": straight,
                                                       "guarded": guarded})
        print("Example 7311", [(row["reason"], row["ticks"], row["progress"])
                               for row in (example, straight, guarded)], flush=True)
    rows = []
    archived = {}
    architecture = "free" if args.free else "symmetric"
    original_name = "outcomes_free" if args.free else "outcomes"
    experiment = "transfer"
    if args.verified_selection:
        experiment = "transfer_verified_selection_free" if args.free else "transfer_verified_selection"
        archived = {row["seed"]: row for row in json.loads((directory / f"{original_name}.json").read_text())}
    for seed in SEEDS:
        genome = weights(seed, architecture, experiment)
        if seed in archived and np.array_equal(genome, weights(seed, architecture)):
            rows.append(archived[seed])
            continue
        row = {"seed": seed}
        for name, course in {"map6": training, **courses}.items():
            outcome = rollout(genome, course, guarded=True, symmetric=not args.free)
            row[name] = {key: value for key, value in outcome.items() if key not in ("path", "commands")}
        rows.append(row)
        print(seed, [(name, row[name]["reason"]) for name in courses], flush=True)
    suffix = "_verified_selection" if args.verified_selection else ""
    save_json(directory / f"{original_name}{suffix}.json", rows)
    print({name: sum(row[name]["reason"] == "finished" for row in rows) for name in courses}, flush=True)


if __name__ == "__main__":
    main()
