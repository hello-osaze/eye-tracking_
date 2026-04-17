# CEC-Gaze Standalone Study

This repository packages our standalone work on **CEC-Gaze** for the EyeBench
claim-verification task (**IITBHGC_CV**). The outer `eye-tracking_` directory
is the hand-in surface; the benchmark code it extends lives under
`source/eyebench`.

## Repository Layout

- `problemsetting.md`: original task description and study requirements
- `cec_study_progress.md`: running log of experiments and intermediate results
- `cec_study_final_report.md`: cleaned study summary
- `cec_workshop_report.md`: workshop-style report draft
- `requirements.txt`: pip-friendly dependency snapshot for the standalone repo
- `submission_assets/`: tables and figures used in the report
- `source/eyebench/`: vendored EyeBench codebase with the CEC-Gaze extensions

## Installation

From the repo root:

```bash
python -m venv source/.venv
./source/.venv/bin/pip install -r requirements.txt
```

If you are targeting NVIDIA GPUs, install the matching CUDA-enabled PyTorch
wheel for your platform instead of the default pip wheel.

## One-Command Run

For the full standalone cloud pipeline, you can now use the repo-level wrapper:

```bash
python run_cec_pipeline.py
```

That runs, in order:

- IITBHGC data download + raw-file union + preprocessing + fold creation + stats
- the direct CEC study and ablations
- the benchmark-facing late-fusion suite
- the faithfulness sweep
- the submission asset rebuild

The default wrapper settings match the official setup we used most often:

- `ROBERTA_LARGE`
- `10` epochs
- unfrozen backbone
- `GPU` accelerator with `1` device
- CUDA faithfulness evaluation
- `4` training workers and `4` evaluation workers
- `32-true` precision by default for result parity

Useful variants:

```bash
python run_cec_pipeline.py --stages data direct fusion
```

```bash
python run_cec_pipeline.py --faithfulness-device cuda --output-tag gpu_large_true_e10
```

```bash
python run_cec_pipeline.py --rerun-existing
```

If your cloud environment already provides a shared results folder such as
`output_data/`, you can point the heavy experiment artifacts there directly:

```bash
python run_cec_pipeline.py --results-root output_data
```

If that storage is a sibling of the repo rather than inside it, use a relative
path that goes up one level:

```bash
python run_cec_pipeline.py --results-root ../output-data
```

For low-storage cloud nodes, move the dataset cache there too:

```bash
python run_cec_pipeline.py \
  --results-root ../output-data \
  --data-root ../output-data/eyebench-data
```

For a cloud node in the rough class of `1x 20 GB GPU / 7 CPU / 70 GB RAM`, the
best first speed knobs are runtime ones rather than model changes:

```bash
python run_cec_pipeline.py \
  --results-root ../output-data \
  --data-root ../output-data/eyebench-data \
  --trainer-num-workers 4 \
  --eval-num-workers 4 \
  --trainer-precision 16-mixed \
  --faithfulness-batch-size 4
```

That keeps the study logic unchanged while making better use of the CPU side of
the machine and the GPU tensor cores.

If your machine already has the processed IITBHGC data and fold files, you can
skip the prep stage:

```bash
python run_cec_pipeline.py --stages direct fusion faithfulness assets
```

The data stage mirrors EyeBench's native `get_data.sh` flow. On a fresh machine
it now checks that the expected processed files exist before moving on to
training, so missing processed IITBHGC artifacts like `ia.feather`,
`fixations.feather`, or `trial_level.feather` surface at the prep step instead
of later inside Hydra training jobs.
If those artifacts already exist, the wrapper skips the `data` stage
automatically unless you pass `--rerun-existing`.

For the late-fusion text baseline, the wrapper now prefers a bundled IITBHGC
Text-Only Roberta raw prediction reference under
`source/eyebench/results/reference_raw_iitbhgc/` before it considers building a
local fallback run. That keeps the cloud pipeline aligned with the study
reference instead of treating a fresh single-config rerun as benchmark-
equivalent.

## Canonical CEC Entry Points

All runnable study code lives in `source/eyebench/src/run/single_run/`.

- `run_iitbhgc_local_benchmark_suite.py`
  Retrains local IITBHGC baselines and CEC ablations under one matched budget.
- `run_cec_extended_late_fusions.py`
  Builds the benchmark-facing late-fusion comparisons against text-only RoBERTa.
- `test_cec_gaze_score_drop.py`
  Runs score-drop perturbation controls.
- `test_cec_gaze_faithfulness.py`
  Runs comprehensiveness and sufficiency faithfulness tests.
- `build_cec_submission_assets.py`
  Rebuilds report tables, figures, and significance summaries.

## Typical Workflow

From the repo root:

```bash
python run_cec_pipeline.py
```

or run the stages manually:

```bash
cd source/eyebench
../.venv/bin/python src/run/single_run/run_iitbhgc_local_benchmark_suite.py \
  --output-root outputs/cec_gaze_ablation_completion_mps_large_true_e10 \
  --models CECGazeNoCoverage CECGazeTextOnly \
  --backbone ROBERTA_LARGE \
  --batch-size 4 \
  --max-epochs 10 \
  --trainer-accelerator GPU \
  --trainer-devices 1 \
  --unfreeze-backbone
```

```bash
cd source/eyebench
../.venv/bin/python src/run/single_run/run_cec_extended_late_fusions.py \
  --cec-root outputs/cec_gaze_claim_context_mlp_mps_large_true_e10 \
  --output-root outputs/cec_roberta_late_fusion_mps_large_true_e10
```

```bash
cd source/eyebench
../.venv/bin/python src/run/single_run/build_cec_submission_assets.py
```

## Notes

- Generated benchmark outputs stay inside `source/eyebench/outputs/`.
- Report-ready figures and tables stay at the outer repo root in
  `submission_assets/`.
- The goal of this layout is to keep the standalone study easy to archive,
  review, and later merge upstream in smaller pieces.
