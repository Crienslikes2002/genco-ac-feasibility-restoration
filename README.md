## Install

Python 3.12. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For inference and timing, install platform-compatible PyTorch, PyG and torch-scatter wheels, then `pip install -r requirements-inference.txt`. For plotting, use `pip install -r requirements-plots.txt`.

## Local inputs

- `data/cases_ieee14.jsonl`, `data/cases_ieee30.jsonl`: exported cases, predictions, references and recorded inference timings.
- `data/frozen_selection.json` and raw graphs at the paths recorded in the exports.
- `models/<grid>/`: `config.yaml`, `normalizer_stats.pt`, `best_model_state_dict.pt`, and the matching `case14_ieee_scenario_splits.json` or `case30_ieee_scenario_splits.json`.

The case schema is defined in `code/ac_physics.py`. Exports also include `case_id`, `grid`, `split`, `scenario_id`, `sample_order`, `prediction`, `reference`, `inference_time_s`, `model_forward_s`, `preprocessing_s`, and `postprocessing_s`. Use seed 42 with 10 validation and 50 test cases per grid for the fixed experiment scripts and plots. Data, checkpoints and generated outputs remain local.

## Run

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Check inference against existing exports; requires local model files.
python code/reinfer_from_bundle.py --output results/reinference_local.jsonl --manifest results/reinference_local_manifest.json

# Evaluate restoration and AC-OPF baselines using saved predictions.
python code/run_experiment.py --split val
python code/run_experiment.py --split test

# Rerun inference and optimization for timing, then generate local plots.
python code/benchmark_runtime.py --repeats 5
python code/make_figures.py
python code/make_runtime_figure.py
```

To run optimization alone on external case exports:

```bash
python code/reproduce_from_export.py --data-dir /path/to/case_exports --split test --output-dir /path/to/new_results
```

Third-party source terms: [GridFM license](vendor/gridfm_graphkit/LICENSE).
