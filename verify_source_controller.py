"""Replay a Figure 9-only controller and check its swept poses on all four tracks."""

import argparse
from hashlib import sha256
import json

import numpy as np

import pilot
from paper_corridors import (
    MATLAB_VEHICLE, ROOT, VehicleSetup, drive, load_corridors, move_vehicle, plot_geometry, safe_step,
    steering_command,
)
from source_reproduction import OPTIONS
from run_experiments import save_json


def interpolated_contacts(genome, course, path, *, symmetric=True, hidden=15, physical_beams=25, noise=None,
                          steering_scale=OPTIONS["steering_scale"], steering_cap=None, projected=False,
                          pose_samples=10, adaptive_speed=False):
    """Check sampled body poses between controller updates."""
    inputs, outputs = pilot.decode(genome, symmetric, hidden)
    position = course.start[None, :].copy()
    heading = np.array([course.heading])
    headings = [heading[0]]
    for tick, endpoint in enumerate(path[1:]):
        sensor_position = position - pilot.HALF_LENGTH * pilot.direction(heading)
        scan = pilot.sense(sensor_position, heading, course, physical_beams=physical_beams,
                          relative_noise=None if noise is None else noise[tick])
        steering = steering_scale * steering_command(scan * OPTIONS["input_scale"], inputs, outputs,
                                                     symmetric, projected)
        if steering_cap is not None:
            steering = np.clip(steering, -steering_cap, steering_cap)
        step = safe_step(MATLAB_VEHICLE.step_distance, scan, steering) if adaptive_speed else MATLAB_VEHICLE.step_distance
        position, heading = move_vehicle(position, heading, steering, step)
        np.testing.assert_allclose(position[0], endpoint, rtol=0, atol=1e-10)
        headings.append(heading[0])

    headings = np.unwrap(headings)
    phases = np.arange(1, pose_samples + 1) / pose_samples
    contacts = 0
    for start in range(0, len(path) - 1, 80):
        end = min(start + 80, len(path) - 1)
        positions = path[start:end, None, :] + phases[None, :, None] * (
            path[start + 1:end + 1] - path[start:end])[:, None, :]
        angles = headings[start:end, None] + phases[None, :] * (
            headings[start + 1:end + 1] - headings[start:end])[:, None]
        angles = angles.ravel()
        body = positions.reshape(-1, 2) + MATLAB_VEHICLE.body_offset * pilot.direction(angles)
        contacts += np.count_nonzero(pilot.collides(body, angles, course))
    return int(contacts)


def replay(genome, course, vehicle):
    options = {**OPTIONS, "vehicle": vehicle}
    _, finished, progress, path = drive(genome, True, course, trace=True, **options)
    return {
        "finished": bool(finished[0]),
        "progress": float(progress[0]),
        "ticks": len(path) - 1,
    }, np.array(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6201)
    args = parser.parse_args()
    directory = ROOT / "results" / "source_alignment"
    weight_path = directory / f"weights_{args.seed}.json"
    frozen = json.loads(weight_path.read_text())
    original = json.loads((directory / f"outcome_{args.seed}.json").read_text())
    if "test" in frozen or frozen["champion"] != original["champion"]:
        raise ValueError("The training archive and test weights differ.")

    genome = np.array([frozen["champion"]])
    courses = load_corridors()
    runs, paths = {}, {}
    for name, course in courses.items():
        outcome, path = replay(genome, course, MATLAB_VEHICLE)
        if name in original["test"] and outcome != original["test"][name]:
            raise ValueError(f"Test replay changed for {name}.")
        outcome["interpolated_contact_poses"] = interpolated_contacts(genome, course, path)
        runs[name] = outcome
        paths[name] = (outcome["finished"], path)
        print(name, outcome, flush=True)

    vehicle_checks = {
        "old_setup": VehicleSetup(),
        "step_only": VehicleSetup(step_distance=1.0),
        "sensor_and_body_reference_only": VehicleSetup(sensor_offset=0.0, body_offset=-pilot.HALF_LENGTH),
    }
    ablation = {
        name: {course_name: replay(genome, course, vehicle)[0] for course_name, course in courses.items()}
        for name, vehicle in vehicle_checks.items()
    }
    output = directory / f"verified_{args.seed}"
    save_json(output.with_suffix(".json"), {
        "weights_sha256": sha256(weight_path.read_bytes()).hexdigest(),
        "training_course": "map6",
        "courses": runs,
        "replay_only_vehicle_checks": ablation,
    })
    plot_geometry(courses, None, output.with_suffix(".png"), trial_paths=paths)


if __name__ == "__main__":
    main()
