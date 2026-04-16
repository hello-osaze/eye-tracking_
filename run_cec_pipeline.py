#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
EYEBENCH_ROOT = REPO_ROOT / 'source' / 'eyebench'
DEFAULT_VENV_PYTHON = REPO_ROOT / 'source' / '.venv' / 'bin' / 'python'
DEFAULT_ROBERTA_ROOT = Path(
    'results/raw/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
    'trainer.wandb_job_type=Roberta_IITBHGC_CV'
)
PIPELINE_STAGES = ['data', 'direct', 'fusion', 'faithfulness', 'assets']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the full standalone CEC-Gaze pipeline: direct study, late fusion, '
            'faithfulness, and submission assets.'
        )
    )
    parser.add_argument(
        '--venv-python',
        type=Path,
        default=DEFAULT_VENV_PYTHON,
        help='Python interpreter inside the project virtualenv.',
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=PIPELINE_STAGES,
        default=PIPELINE_STAGES,
        help='Subset of pipeline stages to run. Default: all.',
    )
    parser.add_argument(
        '--output-tag',
        default='gpu_large_true_e10',
        help=(
            'Suffix used when auto-building output roots. '
            'Example: gpu_large_true_e10.'
        ),
    )
    parser.add_argument(
        '--results-root',
        type=Path,
        default=None,
        help=(
            'Optional parent directory for study outputs. When set, the wrapper '
            'writes the direct and fusion outputs under this folder instead of '
            'source/eyebench/outputs/.'
        ),
    )
    parser.add_argument(
        '--direct-output-root',
        type=Path,
        default=None,
        help='Override the direct-study output root under source/eyebench.',
    )
    parser.add_argument(
        '--fusion-output-root',
        type=Path,
        default=None,
        help='Override the late-fusion output root under source/eyebench.',
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=DEFAULT_ROBERTA_ROOT,
        help='Root containing the raw official RoBERTa prediction dumps.',
    )
    parser.add_argument(
        '--dataset',
        default='IITBHGC',
        help='Dataset name passed to the preprocessing scripts.',
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        help='Fold indices for the direct study.',
    )
    parser.add_argument(
        '--backbone',
        default='ROBERTA_LARGE',
        help='Hydra backbone enum override for the direct study.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Per-step batch size for the direct study.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=10,
        help='Epoch count for the direct study.',
    )
    parser.add_argument(
        '--max-time-limit',
        default=None,
        help='Optional Lightning max_time override for the direct study.',
    )
    parser.add_argument(
        '--trainer-accelerator',
        default='GPU',
        help='Trainer accelerator for the direct study, e.g. GPU or CPU.',
    )
    parser.add_argument(
        '--trainer-devices',
        type=int,
        default=1,
        help='Number of devices for the direct study.',
    )
    parser.add_argument(
        '--trainer-num-workers',
        type=int,
        default=4,
        help='DataLoader workers for training on the direct study.',
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=4,
        help='DataLoader workers for evaluation-heavy stages.',
    )
    parser.add_argument(
        '--trainer-precision',
        default='32-true',
        choices=['32-true', '16-mixed'],
        help='Lightning precision used for training and checkpoint evaluation.',
    )
    parser.add_argument(
        '--wandb-project',
        default='CECGazeFullPipeline',
        help='Offline WandB project label used by the direct-study runner.',
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Keep the text backbone frozen instead of fine-tuning it.',
    )
    parser.add_argument(
        '--drop-fraction',
        type=float,
        default=0.2,
        help='Score-drop token fraction for the direct-study runner.',
    )
    parser.add_argument(
        '--score-drop-random-repeats',
        type=int,
        default=10,
        help='Random-drop repeats for the direct-study runner.',
    )
    parser.add_argument(
        '--alpha-grid-step',
        type=float,
        default=0.001,
        help='Alpha grid step for late fusion.',
    )
    parser.add_argument(
        '--faithfulness-device',
        default='cuda',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        help='Device for the faithfulness evaluation.',
    )
    parser.add_argument(
        '--faithfulness-batch-size',
        type=int,
        default=2,
        help='Batch size for the faithfulness evaluation.',
    )
    parser.add_argument(
        '--faithfulness-fractions',
        nargs='+',
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        help='Token fractions used in the faithfulness sweep.',
    )
    parser.add_argument(
        '--faithfulness-random-repeats',
        type=int,
        default=10,
        help='Random repeats for the faithfulness evaluation.',
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Recompute stages even when their outputs already exist.',
    )
    return parser.parse_args()


def resolve_output_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    def resolve_user_path(path: Path) -> Path:
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    results_root = (
        resolve_user_path(args.results_root) if args.results_root is not None else None
    )
    direct_output_root = args.direct_output_root
    if direct_output_root is None:
        base_root = results_root if results_root is not None else Path('outputs')
        direct_output_root = base_root / f'cec_gaze_claim_context_mlp_{args.output_tag}'
    else:
        direct_output_root = resolve_user_path(direct_output_root)

    fusion_output_root = args.fusion_output_root
    if fusion_output_root is None:
        base_root = results_root if results_root is not None else Path('outputs')
        fusion_output_root = base_root / f'cec_roberta_late_fusion_{args.output_tag}'
    else:
        fusion_output_root = resolve_user_path(fusion_output_root)

    return direct_output_root, fusion_output_root


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault('WANDB_MODE', 'offline')
    env.setdefault('HYDRA_FULL_ERROR', '1')
    env.setdefault('HF_HUB_DISABLE_XET', '1')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    existing_pythonpath = env.get('PYTHONPATH')
    eyebench_pythonpath = str(EYEBENCH_ROOT)
    if existing_pythonpath:
        env['PYTHONPATH'] = os.pathsep.join([eyebench_pythonpath, existing_pythonpath])
    else:
        env['PYTHONPATH'] = eyebench_pythonpath
    return env


def ensure_python_exists(python_bin: Path) -> None:
    if python_bin.exists():
        return
    raise FileNotFoundError(
        f'Python interpreter not found at {python_bin}. '
        'Create the venv first with `python -m venv source/.venv` and install requirements.'
    )


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print()
    print(f'>>> {shlex.join(cmd)}')
    subprocess.run(
        cmd,
        cwd=EYEBENCH_ROOT,
        env=env,
        check=True,
    )


def data_prep_commands(
    args: argparse.Namespace,
    python_bin: Path,
) -> list[list[str]]:
    return [
        [
            str(python_bin),
            'src/data/preprocessing/download_data.py',
            '--dataset',
            args.dataset,
        ],
        [
            str(python_bin),
            'src/data/preprocessing/union_raw_files.py',
            '--dataset',
            args.dataset,
        ],
        [
            str(python_bin),
            'src/data/preprocessing/preprocess_data.py',
            '--dataset',
            args.dataset,
        ],
        [
            str(python_bin),
            'src/data/preprocessing/create_folds.py',
            '--dataset',
            args.dataset,
            '--do_not_recreate_trial_folds',
            '--do_not_recreate_item_subject_folds',
        ],
        [
            str(python_bin),
            'src/data/preprocessing/stats.py',
            '--dataset',
            args.dataset,
        ],
    ]


def expected_dataset_artifacts(dataset: str) -> list[Path]:
    dataset_root = EYEBENCH_ROOT / 'data' / dataset
    return [
        dataset_root / 'precomputed_events' / 'combined_fixations.csv',
        dataset_root / 'processed' / 'fixations.feather',
        dataset_root / 'processed' / 'ia.feather',
        dataset_root / 'processed' / 'trial_level.feather',
    ]


def ensure_dataset_prepared(dataset: str) -> None:
    missing = [path for path in expected_dataset_artifacts(dataset) if not path.exists()]
    if not missing:
        return
    missing_text = '\n'.join(f'  - {path.relative_to(EYEBENCH_ROOT)}' for path in missing)
    raise FileNotFoundError(
        'Dataset preparation did not produce the expected EyeBench artifacts.\n'
        f'Missing files:\n{missing_text}\n'
        'The most common cause is an incomplete raw-data preparation step.'
    )


def direct_study_command(
    args: argparse.Namespace,
    python_bin: Path,
    direct_output_root: Path,
) -> list[str]:
    cmd = [
        str(python_bin),
        'src/run/single_run/run_cec_gaze_full_study.py',
        '--output-root',
        str(direct_output_root),
        '--folds',
        *[str(fold) for fold in args.folds],
        '--backbone',
        args.backbone,
        '--batch-size',
        str(args.batch_size),
        '--max-epochs',
        str(args.max_epochs),
        '--trainer-accelerator',
        args.trainer_accelerator,
        '--trainer-devices',
        str(args.trainer_devices),
        '--trainer-num-workers',
        str(args.trainer_num_workers),
        '--eval-num-workers',
        str(args.eval_num_workers),
        '--trainer-precision',
        args.trainer_precision,
        '--wandb-project',
        args.wandb_project,
        '--drop-fraction',
        str(args.drop_fraction),
        '--random-repeats',
        str(args.score_drop_random_repeats),
    ]
    if args.max_time_limit is not None:
        cmd.extend(['--max-time-limit', args.max_time_limit])
    if not args.freeze_backbone:
        cmd.append('--unfreeze-backbone')
    if args.rerun_existing:
        cmd.append('--rerun-existing')
    return cmd


def late_fusion_command(
    args: argparse.Namespace,
    python_bin: Path,
    direct_output_root: Path,
    fusion_output_root: Path,
) -> list[str]:
    cmd = [
        str(python_bin),
        'src/run/single_run/run_cec_extended_late_fusions.py',
        '--roberta-root',
        str(args.roberta_root),
        '--cec-root',
        str(direct_output_root),
        '--output-root',
        str(fusion_output_root),
        '--alpha-grid-step',
        str(args.alpha_grid_step),
    ]
    if args.rerun_existing:
        cmd.append('--rerun-existing')
    return cmd


def faithfulness_command(
    args: argparse.Namespace,
    python_bin: Path,
    direct_output_root: Path,
) -> list[str]:
    return [
        str(python_bin),
        'src/run/single_run/test_cec_gaze_faithfulness.py',
        '--eval-path',
        str(direct_output_root / 'CECGaze'),
        '--device',
        args.faithfulness_device,
        '--batch-size',
        str(args.faithfulness_batch_size),
        '--num-workers',
        str(args.eval_num_workers),
        '--eval-types',
        'test',
        '--fractions',
        *[str(fraction) for fraction in args.faithfulness_fractions],
        '--random-repeats',
        str(args.faithfulness_random_repeats),
    ]


def assets_command(
    args: argparse.Namespace,
    python_bin: Path,
    direct_output_root: Path,
    fusion_output_root: Path,
) -> list[str]:
    return [
        str(python_bin),
        'src/run/single_run/build_cec_submission_assets.py',
        '--roberta-root',
        str(args.roberta_root),
        '--direct-root-main',
        str(direct_output_root),
        '--direct-root-ablation',
        str(direct_output_root),
        '--fusion-root',
        str(fusion_output_root),
    ]


def main() -> int:
    args = parse_args()
    python_bin = args.venv_python.resolve()
    direct_output_root, fusion_output_root = resolve_output_roots(args=args)
    env = build_env()

    ensure_python_exists(python_bin=python_bin)
    if not EYEBENCH_ROOT.exists():
        raise FileNotFoundError(f'EyeBench root not found at {EYEBENCH_ROOT}')

    ordered_stages = [stage for stage in PIPELINE_STAGES if stage in set(args.stages)]

    print('Running CEC-Gaze pipeline with:')
    print(f'  stages: {", ".join(ordered_stages)}')
    print(f'  python: {python_bin}')
    print(f'  direct outputs: {direct_output_root}')
    print(f'  fusion outputs: {fusion_output_root}')
    print(f'  faithfulness device: {args.faithfulness_device}')
    print(f'  trainer workers: {args.trainer_num_workers}')
    print(f'  eval workers: {args.eval_num_workers}')
    print(f'  trainer precision: {args.trainer_precision}')

    commands = {
        'direct': direct_study_command(
            args=args,
            python_bin=python_bin,
            direct_output_root=direct_output_root,
        ),
        'fusion': late_fusion_command(
            args=args,
            python_bin=python_bin,
            direct_output_root=direct_output_root,
            fusion_output_root=fusion_output_root,
        ),
        'faithfulness': faithfulness_command(
            args=args,
            python_bin=python_bin,
            direct_output_root=direct_output_root,
        ),
        'assets': assets_command(
            args=args,
            python_bin=python_bin,
            direct_output_root=direct_output_root,
            fusion_output_root=fusion_output_root,
        ),
    }

    for stage in ordered_stages:
        print()
        print(f'=== Stage: {stage} ===')
        if stage == 'data':
            for cmd in data_prep_commands(args=args, python_bin=python_bin):
                run_command(cmd=cmd, env=env)
            ensure_dataset_prepared(dataset=args.dataset)
            continue
        run_command(cmd=commands[stage], env=env)

    print()
    print('CEC-Gaze pipeline finished successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
