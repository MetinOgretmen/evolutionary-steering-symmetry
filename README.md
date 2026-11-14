# Evolutionary Steering Controllers

Python simulation and neuroevolution tools for steering a rectangular vehicle from front-facing range measurements. The code compares paired-input symmetric, unrestricted, smaller unrestricted and output-symmetric neural networks using GA and CMA-ES.

## Code

| Files | Function |
| --- | --- |
| `pilot.py`, `paper_corridors.py`, `corridor_search.py` | Vehicle motion, range sensing, collisions, course geometry and compiled evaluation |
| `transfer_experiment.py`, `followup_cma.py` | Single-course training, controller selection and frozen-controller evaluation |
| `followup_initial.py`, `followup_mutation.py` | Initial-weight sweeps, output projection and mutation-rate controls |
| `followup_courses.py`, `single_course_corners.py` | Generated corner courses and corner-focused training |
| `source_reproduction.py`, `run_experiments.py` | Sensor, noise and evolutionary-setting comparisons |
| `analyze_*.py`, `audit_*.py`, `summarize_*.py`, `verify_*.py` | Replay validation, steering probes, clearance measurements and numerical summaries |
| `footprint_guard.py`, `fresh_guard_check.py`, `corridor_holdout.py` | Optional body-aware steering intervention and additional course evaluations |
| `plot_figures.py` | Course trajectories, weight-scale plots and geometry plots |
| `experiment_config.json`, `paper_*_mask*.npz` | Experiment settings and course masks |
| `data/reference_runs.zip`, `prepare_data.py` | Saved controller weights and numerical outputs, restored into `results/` |

The optional body-aware intervention is separate from the unassisted controller evaluations.

## Run

Use Python 3.13. Install the dependencies, restore the saved outputs and run the validation:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python verify_followup.py
```

Small GA and CMA-ES training checks:

```bash
python transfer_experiment.py --smoke
python followup_cma.py --smoke
```

Generate plots with `python plot_figures.py outputs`. Add `--courses-only` for the course overview. Each script lists its arguments with `--help`; image-based diagnostics also accept a source image path.

## Sources

The symmetry model follows [Eraqi, Nagiub and Sidra (2022)](https://arxiv.org/abs/2203.15522). The vehicle model draws on the earlier [GA-NN-Car MATLAB implementation](https://github.com/heshameraqi/GA-NN-Car/tree/b68301d).
