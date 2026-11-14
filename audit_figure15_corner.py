"""Diagnose a Figure 15 contact cluster without changing controller commands."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

import pilot
from audit_steering import alternate_figure15
from footprint_guard import PHASES, VEHICLE, between, rollout, weights
from fresh_guard_check import ARCHIVES
from paper_corridors import ROOT, SCALE, load_corridors, move_vehicle
from run_experiments import save_json
from transfer_experiment import SEEDS


WINDOW = np.array([[9, -172], [29, -156]])
PROTRUSION = np.array([[17.5, -164.5], [21.5, -158.5]])
TURN_THRESHOLDS_DEG = (15, 20, 25)


def in_window(points, window):
    return np.all((points >= window[0]) & (points <= window[1]), axis=-1)


def near_boxes(course, window):
    boxes = course.boxes.reshape(-1, 2, 2)
    return boxes[np.all((boxes[:, 0] <= window[1]) & (boxes[:, 1] >= window[0]), axis=-1)]


def protrusion_segments(course):
    candidates = near_boxes(course, PROTRUSION)
    return candidates[in_window(candidates, PROTRUSION).all(axis=1)]


def footprint_margin(body, heading, boxes):
    """Signed rectangle-to-segment margin. Negative is the pairwise SAT overlap."""
    forward = pilot.direction(heading)
    side = pilot.direction(heading + np.pi / 2)
    axes = np.vstack((np.eye(2), forward, side))
    half_length, half_width = pilot.HALF_LENGTH, pilot.HALF_WIDTH
    radii = half_length * np.abs(axes @ forward) + half_width * np.abs(axes @ side)
    projections = (boxes - body) @ axes.T
    gaps = np.maximum(projections.min(axis=1), -projections.max(axis=1)) - radii
    sat = gaps.max(axis=1)
    if np.any(sat <= 0):
        index = int(sat.argmin())
        return float(sat[index]), index

    corners = body + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) @ np.vstack(
        (half_length * forward, half_width * side))
    endpoint_local = (boxes - body) @ np.vstack((forward, side)).T
    endpoint_gaps = np.maximum(0, np.abs(endpoint_local) - [half_length, half_width])
    endpoint_distances = np.linalg.norm(endpoint_gaps, axis=-1).min(axis=1)
    edges = boxes[:, 1] - boxes[:, 0]
    offsets = corners[None] - boxes[:, :1]
    fractions = np.clip(np.einsum("nki,ni->nk", offsets, edges) / np.sum(edges**2, axis=1)[:, None], 0, 1)
    corner_distances = np.linalg.norm(offsets - fractions[..., None] * edges[:, None], axis=-1).min(axis=1)
    distances = np.minimum(corner_distances, endpoint_distances)
    index = int(distances.argmin())
    return float(distances[index]), index


def contact_span(body, heading, box):
    """Longitudinal position of the wall part inside the body, relative to its centre."""
    axes = np.vstack((pilot.direction(heading), pilot.direction(heading + np.pi / 2)))
    start, end = (box - body) @ axes.T
    lower, upper = 0.0, 1.0
    for coordinate, step, half in zip(start, end - start, (pilot.HALF_LENGTH, pilot.HALF_WIDTH)):
        if abs(step) < 1e-12:
            if abs(coordinate) > half + 1e-9:
                return None
            continue
        limits = sorted(((-half - coordinate) / step, (half - coordinate) / step))
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
    if lower > upper + 1e-9:
        return None
    return sorted((start[0] + lower * (end[0] - start[0]), start[0] + upper * (end[0] - start[0])))


def sampled_path(run, course):
    path = np.asarray(run["path"])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    heading = np.array([course.heading])
    frames = []
    for index, (start, end, command) in enumerate(zip(path[:-1], path[1:], run["commands"])):
        reconstructed, next_heading = move_vehicle(start[None], heading,
                                                    np.deg2rad([command[1]]), VEHICLE.step_distance)
        np.testing.assert_allclose(reconstructed[0], end, atol=1e-9, rtol=0)
        if in_window(start, WINDOW) or in_window(end, WINDOW):
            bodies, angles = between(start[None], heading, end[None], next_heading, PHASES)
            distance = arc[index] + PHASES * (arc[index + 1] - arc[index])
            frames.append((index, start, heading[0], command[0], arc[index], bodies[0], angles[0], distance))
        heading = next_heading
    return frames


def corner_analysis(frames, sensor_course, geometries):
    boxes = {name: near_boxes(course, WINDOW + [[-5, -5], [5, 5]]) for name, course in geometries.items()}
    target = protrusion_segments(sensor_course)
    pose_rows, commands = [], []
    for index, start, heading, steering, arc, bodies, angles, distances in frames:
        scan = pilot.sense((start - pilot.HALF_LENGTH * pilot.direction(heading))[None], np.array([heading]),
                           sensor_course)[0] * pilot.SENSOR_RANGE
        points = start + scan[:, None] * pilot.direction(heading + pilot.BEAM_ANGLES)
        hits = ((points[:, None] >= target[None, :, 0] - 1e-8)
                & (points[:, None] <= target[None, :, 1] + 1e-8)).all(axis=-1)
        commands.append({"step": index + 1, "s_m": arc, "steering_deg": steering,
                         "protrusion_rays": int(np.count_nonzero(hits.any(axis=1) & (scan < pilot.SENSOR_RANGE)))})
        for phase, (body, angle, distance) in enumerate(zip(bodies, angles, distances)):
            margins = {}
            for name, walls in boxes.items():
                margin, wall = footprint_margin(body, angle, walls)
                record = {"margin_m": margin}
                if margin <= 0:
                    record["wall_m"] = walls[wall].tolist()
                    record["contact_longitudinal_m"] = contact_span(body, angle, walls[wall])
                margins[name] = record
            pose_rows.append({"step": index + 1, "phase": float(PHASES[phase]), "s_m": distance,
                              "body_m": body.tolist(), "margins": margins})
    return commands, pose_rows


def summarize_local(commands, poses, source_name, geometries):
    if not poses:
        return None
    hits = [step for step in commands if step["protrusion_rays"]]
    last = hits[-1] if hits else None
    loss = next((step for step in commands if last and step["step"] == last["step"] + 1), None)
    turns = {str(threshold): next((step for step in commands if step["steering_deg"] <= -threshold), None)
             for threshold in TURN_THRESHOLDS_DEG}
    return {
        "last_protrusion_hit": last,
        "first_step_after_last_hit": loss,
        "first_right_turn_at_threshold": turns,
        "s_turn_minus_loss_m": {threshold: turn["s_m"] - loss["s_m"] if loss and turn else None
                                for threshold, turn in turns.items()},
        "same_path": {name: {
            "min_margin_m": min(pose["margins"][name]["margin_m"] for pose in poses),
            "first_contact": next((pose for pose in poses if pose["margins"][name]["margin_m"] <= 0), None),
        } for name in geometries},
        "path_ended_at_corner_contact": poses[-1]["margins"][source_name]["margin_m"] <= 0
            or any(pose["margins"][source_name]["margin_m"] <= 0 for pose in poses
                   if pose["step"] == poses[-1]["step"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure15_image", type=Path, help="Original embedded 390×499 Figure 15 image")
    args = parser.parse_args()
    original = load_corridors()["map9"]
    alternate = alternate_figure15(original, args.figure15_image)
    geometries = {"original": original, "alternate": alternate}
    rows = []
    for architecture, archive in ARCHIVES.items():
        for seed in SEEDS:
            genome = weights(seed, architecture, archive)
            for name, course in geometries.items():
                run = rollout(genome, course, symmetric=architecture == "symmetric")
                assert run["interventions"] == run["short_steps"] == run["goal_steps"] == 0
                frames = sampled_path(run, course)
                commands, poses = corner_analysis(frames, course, geometries)
                summary = summarize_local(commands, poses, name, geometries)
                if summary and summary["path_ended_at_corner_contact"]:
                    assert run["reason"] == "body_contact"
                    assert summary["same_path"][name]["first_contact"]["step"] == run["ticks"]
                rows.append({"architecture": architecture, "seed": seed, "source_geometry": name,
                             "reason": run["reason"], "progress": run["progress"], "ticks": run["ticks"],
                             "weight_sha256": sha256(genome.tobytes()).hexdigest(), "local": summary,
                             "local_commands": commands, "local_poses": poses})
    directory = ROOT / "results" / "figure15_corner_diagnosis"
    directory.mkdir(exist_ok=True)
    save_json(directory / "replays.json", {
        "image_sha256": sha256(args.figure15_image.read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "meters_per_pixel": SCALE, "window_m": WINDOW.tolist(),
        "protrusion_window_m": PROTRUSION.tolist(), "turn_thresholds_deg": TURN_THRESHOLDS_DEG,
        "masks": {name: {"free_pixels": int(course.mask.sum()), "sha256": sha256(course.mask.tobytes()).hexdigest(),
                         "protrusion_segments_m": protrusion_segments(course).tolist()}
                  for name, course in geometries.items()},
        "rows": rows,
    })
    for name in geometries:
        for architecture in ARCHIVES:
            subset = [row for row in rows if row["architecture"] == architecture and row["source_geometry"] == name]
            print(name, architecture, "finished", sum(row["reason"] == "finished" for row in subset),
                  "reached", sum(row["local"] is not None for row in subset))
    for row in rows:
        local = row["local"]
        if row["source_geometry"] == "original" and local and local["path_ended_at_corner_contact"]:
            print(row["architecture"], row["seed"],
                  {name: round(data["min_margin_m"], 3) for name, data in local["same_path"].items()},
                  local["s_turn_minus_loss_m"], local["same_path"]["original"]["first_contact"])


if __name__ == "__main__":
    main()
