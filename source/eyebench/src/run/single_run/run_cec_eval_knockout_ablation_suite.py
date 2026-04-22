from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = REPO_ROOT.parent / '.venv' / 'bin' / 'python'


def normalize_trainer_accelerator(accelerator: str) -> str:
    lowered = accelerator.strip().lower()
    if lowered == 'gpu':
        return 'cuda'
    return lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run fast evaluation-time knockout ablations for a trained CEC checkpoint root.'
    )
    parser.add_argument(
        '--cec-root',
        type=Path,
        default=Path('outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGaze'),
        help='Root containing learned CEC fold checkpoints and prediction CSVs.',
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=Path(
            'results/raw/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
            'trainer.wandb_job_type=Roberta_IITBHGC_CV'
        ),
        help='Root containing the official raw RoBERTa predictions.',
    )
    parser.add_argument(
        '--fusion-output-root',
        type=Path,
        default=Path('outputs/cec_roberta_late_fusion_mps_large_true_e10'),
        help='Root under which late-fusion outputs should be stored.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Evaluation batch size override.',
    )
    parser.add_argument(
        '--trainer-accelerator',
        default='GPU',
        help='Accelerator override passed through to test_dl.py.',
    )
    parser.add_argument(
        '--trainer-devices',
        type=int,
        default=1,
        help='Device count override passed through to test_dl.py.',
    )
    parser.add_argument(
        '--alpha-grid-step',
        type=float,
        default=0.001,
        help='Grid step for late-fusion alpha tuning.',
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Recompute evaluation CSVs and fusion outputs even when they already exist.',
    )
    parser.add_argument(
        '--permutation-seed',
        type=int,
        default=42,
        help='Seed used by evaluation-time gaze permutations.',
    )
    return parser.parse_args()


def permutation_result_filename(mode: str, seed: int) -> str:
    return f'trial_level_test_results_gazeperm_{mode}_seed_{seed}.csv'


def variants(args: argparse.Namespace) -> list[dict[str, str | list[str]]]:
    base_variant_name = args.cec_root.name
    specs: list[dict[str, str | list[str]]] = []
    if 'NoCoverage' not in base_variant_name:
        specs.append(
            {
                'label': 'zero_coverage',
                'eval_overrides': ['+model.eval_zero_coverage=true'],
                'result_filename': 'trial_level_test_results_zero_coverage.csv',
                'fusion_output_subdir': f'{base_variant_name}ZeroCoverageRobertaValBlendFine',
                'fusion_model_name': (
                    f'{base_variant_name}ZeroCoverage+RoBERTaValBlendFine-true-e10'
                ),
            }
        )
    specs.extend(
        [
            {
                'label': 'zero_gaze',
                'eval_overrides': ['+model.eval_zero_gaze_features=true'],
                'result_filename': 'trial_level_test_results_zero_gaze.csv',
                'fusion_output_subdir': f'{base_variant_name}ZeroGazeRobertaValBlendFine',
                'fusion_model_name': (
                    f'{base_variant_name}ZeroGaze+RoBERTaValBlendFine-true-e10'
                ),
            },
            {
                'label': 'within_paragraph_permutation',
                'eval_overrides': [
                    '+model.eval_gaze_permutation_mode=within_paragraph',
                    f'+model.eval_gaze_permutation_seed={args.permutation_seed}',
                ],
                'result_filename': permutation_result_filename(
                    mode='within_paragraph',
                    seed=args.permutation_seed,
                ),
                'fusion_output_subdir': (
                    f'{base_variant_name}WithinParagraphPermRobertaValBlendFine'
                ),
                'fusion_model_name': (
                    f'{base_variant_name}WithinParagraphPerm+RoBERTaValBlendFine-true-e10'
                ),
            },
            {
                'label': 'across_participants_permutation',
                'eval_overrides': [
                    '+model.eval_gaze_permutation_mode=across_participants',
                    f'+model.eval_gaze_permutation_seed={args.permutation_seed}',
                ],
                'result_filename': permutation_result_filename(
                    mode='across_participants',
                    seed=args.permutation_seed,
                ),
                'fusion_output_subdir': (
                    f'{base_variant_name}AcrossParticipantsPermRobertaValBlendFine'
                ),
                'fusion_model_name': (
                    f'{base_variant_name}AcrossParticipantsPerm+RoBERTaValBlendFine-true-e10'
                ),
            },
            {
                'label': 'across_labels_permutation',
                'eval_overrides': [
                    '+model.eval_gaze_permutation_mode=across_labels',
                    f'+model.eval_gaze_permutation_seed={args.permutation_seed}',
                ],
                'result_filename': permutation_result_filename(
                    mode='across_labels',
                    seed=args.permutation_seed,
                ),
                'fusion_output_subdir': (
                    f'{base_variant_name}AcrossLabelsPermRobertaValBlendFine'
                ),
                'fusion_model_name': (
                    f'{base_variant_name}AcrossLabelsPerm+RoBERTaValBlendFine-true-e10'
                ),
            },
        ]
    )
    return specs


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    logger.info('Running: {}', ' '.join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def eval_outputs_exist(cec_root: Path, result_filename: str) -> bool:
    return all(
        (cec_root / f'fold_index={fold_index}' / result_filename).exists()
        for fold_index in [0, 1, 2, 3]
    )


def run_eval_ablation(
    args: argparse.Namespace,
    env: dict[str, str],
    spec: dict[str, str | list[str]],
) -> None:
    result_filename = str(spec['result_filename'])
    if eval_outputs_exist(args.cec_root, result_filename) and not args.rerun_existing:
        logger.info('Skipping {} eval: outputs already exist', spec['label'])
        return

    cmd = [
        str(PYTHON_BIN),
        'src/run/single_run/test_dl.py',
        f'eval_path={args.cec_root}',
        f'model.batch_size={args.batch_size}',
        f'+trainer.accelerator={normalize_trainer_accelerator(args.trainer_accelerator)}',
        f'trainer.devices={args.trainer_devices}',
        *[str(override) for override in spec['eval_overrides']],
    ]
    run_command(cmd=cmd, env=env)


def run_late_fusion(
    args: argparse.Namespace,
    env: dict[str, str],
    spec: dict[str, str | list[str]],
) -> None:
    output_root = args.fusion_output_root / str(spec['fusion_output_subdir'])
    summary_path = output_root / 'summary' / 'summary_metrics.csv'
    if summary_path.exists() and not args.rerun_existing:
        logger.info('Skipping {} fusion: summary already exists', spec['label'])
        return

    cmd = [
        str(PYTHON_BIN),
        'src/run/single_run/run_cec_roberta_late_fusion.py',
        '--roberta-root',
        str(args.roberta_root),
        '--cec-root',
        str(args.cec_root),
        '--cec-result-filename',
        str(spec['result_filename']),
        '--output-root',
        str(output_root),
        '--model-name',
        str(spec['fusion_model_name']),
        '--alpha-grid-step',
        str(args.alpha_grid_step),
    ]
    run_command(cmd=cmd, env=env)


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['HYDRA_FULL_ERROR'] = '1'
    env['HF_HUB_DISABLE_XET'] = '1'
    env['MPLCONFIGDIR'] = '/tmp/matplotlib-eyebench'

    for spec in variants(args):
        run_eval_ablation(args=args, env=env, spec=spec)
        run_late_fusion(args=args, env=env, spec=spec)


if __name__ == '__main__':
    main()
