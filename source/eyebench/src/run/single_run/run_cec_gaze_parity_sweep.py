from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from src.run.single_run.report_tables import safe_to_markdown
from src.run.single_run.run_cec_gaze_full_study import (
    StudyPaths,
    ensure_paths,
    evaluate_model,
    fold_dir,
    has_checkpoint,
    hydra_precision_name,
    load_reference_baselines,
    run_command,
    run_score_drop,
    summarize_models,
    summarize_score_drop,
    summarize_thresholds,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = REPO_ROOT.parent / '.venv' / 'bin' / 'python'
ABLATION_MODELS = [
    'CECGazeNoScorer',
    'CECGazeNoCoverage',
    'CECGazeTextOnly',
]


@dataclass(frozen=True)
class CandidateConfig:
    learning_rate: float
    freeze_backbone: bool
    dropout_prob: float

    @property
    def tag(self) -> str:
        freeze_text = 'freeze' if self.freeze_backbone else 'unfreeze'
        lr_text = f'{self.learning_rate:.0e}'.replace('+0', '').replace('+', '')
        dropout_text = str(self.dropout_prob).replace('.', 'p')
        return f'lr_{lr_text}__{freeze_text}__dropout_{dropout_text}'

    @property
    def display_name(self) -> str:
        return (
            f'lr={self.learning_rate:.0e}, '
            f'freeze_backbone={self.freeze_backbone}, '
            f'dropout={self.dropout_prob:g}'
        )


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {'true', '1', 'yes', 'y'}:
        return True
    if lowered in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(
        f'Expected a boolean value like true/false, got {value!r}.'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run a benchmark-parity CEC-Gaze sweep, select the best config by '
            'validation AUROC, and materialize only the winning checkpoints.'
        )
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('outputs/cec_gaze_parity_sweep_cpu'),
        help='Directory where the selected CEC fold checkpoints and summary files are stored.',
    )
    parser.add_argument(
        '--sweep-root',
        type=Path,
        default=None,
        help=(
            'Optional directory for intermediate sweep candidates. Defaults to '
            '<output-root>/_cec_sweep_candidates.'
        ),
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        help='Fold indices to train/evaluate.',
    )
    parser.add_argument(
        '--backbone',
        default='ROBERTA_LARGE',
        help='Hydra backbone enum override, e.g. ROBERTA_BASE or ROBERTA_LARGE.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Per-step batch size override.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=10,
        help='Number of epochs per fold.',
    )
    parser.add_argument(
        '--max-time-limit',
        default=None,
        help=(
            'Optional Lightning max_time override per fold, formatted as DD:HH:MM:SS. '
            'When omitted, the model default time budget is used.'
        ),
    )
    parser.add_argument(
        '--trainer-accelerator',
        default='GPU',
        help='Hydra trainer accelerator enum override, e.g. CPU or GPU.',
    )
    parser.add_argument(
        '--trainer-devices',
        type=int,
        default=1,
        help='Number of devices passed to the trainer.',
    )
    parser.add_argument(
        '--trainer-num-workers',
        type=int,
        default=4,
        help='DataLoader workers used during training.',
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=4,
        help='DataLoader workers used during checkpoint evaluation and score-drop.',
    )
    parser.add_argument(
        '--trainer-precision',
        default='32-true',
        choices=['32-true', '16-mixed'],
        help='Lightning precision setting for training and checkpoint evaluation.',
    )
    parser.add_argument(
        '--wandb-project',
        default='CECGazeParitySweep',
        help='Offline WandB project name used by the sweep runner.',
    )
    parser.add_argument(
        '--drop-fraction',
        type=float,
        default=0.2,
        help='Fraction of highest-weighted context tokens to remove in score-drop.',
    )
    parser.add_argument(
        '--random-repeats',
        type=int,
        default=10,
        help='Number of random-drop repeats for score-drop.',
    )
    parser.add_argument(
        '--learning-rate-values',
        nargs='+',
        type=float,
        default=[1e-5, 3e-5, 1e-4],
        help='Trainer learning-rate grid. Mirrors the EyeBench Roberta-family sweep.',
    )
    parser.add_argument(
        '--freeze-backbone-values',
        nargs='+',
        type=parse_bool,
        default=[True, False],
        help='Freeze/unfreeze grid for the text backbone.',
    )
    parser.add_argument(
        '--dropout-values',
        nargs='+',
        type=float,
        default=[0.1, 0.3, 0.5],
        help='CEC dropout grid. Mirrors the EyeBench Roberta-family dropout sweep.',
    )
    parser.add_argument(
        '--lambda-gold',
        type=float,
        default=0.0,
        help='Auxiliary gold-label loss weight for CEC training.',
    )
    parser.add_argument(
        '--lambda-annotator',
        type=float,
        default=0.0,
        help='Auxiliary annotator-label loss weight for CEC training.',
    )
    parser.add_argument(
        '--lambda-sparse',
        type=float,
        default=0.0,
        help='Sparse evidence regularization weight for CEC training.',
    )
    parser.add_argument(
        '--keep-all-candidate-artifacts',
        action='store_true',
        help='Keep every candidate checkpoint directory instead of pruning non-best runs.',
    )
    parser.add_argument(
        '--keep-final-wandb-offline-runs',
        action='store_true',
        help='Keep the offline WandB folders inside the selected fold directories.',
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Retrain/evaluate even if expected artifacts already exist.',
    )
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['EYEBENCH_DISABLE_WANDB_MEDIA'] = '1'
    env['HYDRA_FULL_ERROR'] = '1'
    env['HF_HUB_DISABLE_XET'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    return env


def sweep_root(output_root: Path, requested_sweep_root: Path | None) -> Path:
    return requested_sweep_root if requested_sweep_root is not None else output_root / '_cec_sweep_candidates'


def candidate_root(base_sweep_root: Path, candidate: CandidateConfig) -> Path:
    return base_sweep_root / candidate.tag


def final_outputs_exist(
    output_root: Path,
    folds: list[int],
    args: argparse.Namespace,
) -> bool:
    best_config_path = output_root / 'summary' / 'best_config.json'
    if not best_config_path.exists():
        return False
    try:
        best_config = json.loads(best_config_path.read_text())
    except json.JSONDecodeError:
        return False
    if best_config.get('selection_protocol') != 'validation_only':
        return False
    expected_lambdas = {
        'lambda_gold': float(args.lambda_gold),
        'lambda_annotator': float(args.lambda_annotator),
        'lambda_sparse': float(args.lambda_sparse),
    }
    for key, expected_value in expected_lambdas.items():
        if float(best_config.get(key, 0.0)) != expected_value:
            return False
    return all(
        (output_root / 'CECGaze' / f'fold_index={fold}' / 'trial_level_test_results.csv').exists()
        for fold in folds
    )


def ablation_outputs_exist(output_root: Path, folds: list[int]) -> bool:
    return all(
        (
            output_root / model_name / f'fold_index={fold}' / 'trial_level_test_results.csv'
        ).exists()
        for model_name in ABLATION_MODELS
        for fold in folds
    )


def candidate_outputs_exist(
    candidate_root_path: Path,
    args: argparse.Namespace,
) -> bool:
    summary_path = candidate_root_path / 'summary' / 'cec_gaze_summary_metrics.csv'
    metrics_path = candidate_root_path / 'summary' / 'candidate_metrics.json'
    if not (summary_path.exists() and metrics_path.exists()):
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    expected_lambdas = {
        'lambda_gold': float(args.lambda_gold),
        'lambda_annotator': float(args.lambda_annotator),
        'lambda_sparse': float(args.lambda_sparse),
    }
    return all(float(metrics.get(key, float('nan'))) == value for key, value in expected_lambdas.items())


def prune_candidate_artifacts(candidate_root_path: Path) -> None:
    model_root = candidate_root_path / 'CECGaze'
    if model_root.exists():
        logger.info('Pruning candidate artifacts under {}', model_root)
        shutil.rmtree(model_root)


def remove_wandb_runs(model_root: Path) -> None:
    for wandb_dir in model_root.glob('fold_index=*/wandb'):
        if wandb_dir.exists():
            shutil.rmtree(wandb_dir)


def remove_fold_checkpoints(model_root: Path, fold_index: int) -> None:
    fold_root = model_root / f'fold_index={fold_index}'
    for checkpoint_path in fold_root.glob('*.ckpt'):
        checkpoint_path.unlink()


def candidate_grid(args: argparse.Namespace) -> list[CandidateConfig]:
    seen: set[tuple[float, bool, float]] = set()
    configs = []
    for learning_rate, freeze_backbone, dropout_prob in itertools.product(
        args.learning_rate_values,
        args.freeze_backbone_values,
        args.dropout_values,
    ):
        key = (float(learning_rate), bool(freeze_backbone), float(dropout_prob))
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            CandidateConfig(
                learning_rate=float(learning_rate),
                freeze_backbone=bool(freeze_backbone),
                dropout_prob=float(dropout_prob),
            )
        )
    return configs


def train_model_fold(
    paths: StudyPaths,
    model_name: str,
    learning_rate: float,
    freeze_backbone: bool,
    dropout_prob: float,
    run_tag: str,
    fold_index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    current_fold_dir = fold_dir(paths.output_root, model_name, fold_index)
    if has_checkpoint(current_fold_dir) and not args.rerun_existing:
        logger.info(
            'Skipping train {} fold {}: checkpoint already exists',
            run_tag,
            fold_index,
        )
        return

    hydra_run_dir = str(current_fold_dir).replace('=', r'\=')
    cmd = [
        str(PYTHON_BIN),
        'src/run/single_run/train.py',
        '+data=IITBHGC_CV',
        f'+model={model_name}',
        '+trainer=TrainerDL',
        f'data.fold_index={fold_index}',
        'trainer.run_mode=TRAIN',
        f'trainer.accelerator={args.trainer_accelerator}',
        f'trainer.devices={args.trainer_devices}',
        f'trainer.num_workers={args.trainer_num_workers}',
        f'trainer.precision={hydra_precision_name(args.trainer_precision)}',
        f'trainer.wandb_project={args.wandb_project}',
        f'trainer.wandb_job_type={run_tag}_fold{fold_index}',
        f'trainer.learning_rate={learning_rate}',
        f'model.backbone={args.backbone}',
        f'model.batch_size={args.batch_size}',
        'model.accumulate_grad_batches=1',
        f'model.max_epochs={args.max_epochs}',
        f'model.freeze_backbone={str(freeze_backbone).lower()}',
        f'model.dropout_prob={dropout_prob}',
        f'model.lambda_gold={args.lambda_gold}',
        f'model.lambda_annotator={args.lambda_annotator}',
        f'model.lambda_sparse={args.lambda_sparse}',
        *(
            [f'model.max_time_limit={args.max_time_limit}']
            if args.max_time_limit is not None
            else []
        ),
        f'hydra.run.dir={hydra_run_dir}',
    ]
    run_command(
        cmd=cmd,
        log_path=paths.logs_root / f'train_{run_tag}_fold_{fold_index}.log',
        env=env,
    )


def train_candidate_fold(
    paths: StudyPaths,
    candidate: CandidateConfig,
    fold_index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    train_model_fold(
        paths=paths,
        model_name='CECGaze',
        learning_rate=candidate.learning_rate,
        freeze_backbone=candidate.freeze_backbone,
        dropout_prob=candidate.dropout_prob,
        run_tag=candidate.tag,
        fold_index=fold_index,
        args=args,
        env=env,
    )


def materialize_main_model(
    output_root: Path,
    candidate: CandidateConfig,
    folds: list[int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    final_paths = ensure_paths(output_root=output_root)
    logger.info(
        'Retraining winning CECGaze config {} into canonical output root',
        candidate.display_name,
    )
    for fold_index in folds:
        train_model_fold(
            paths=final_paths,
            model_name='CECGaze',
            learning_rate=candidate.learning_rate,
            freeze_backbone=candidate.freeze_backbone,
            dropout_prob=candidate.dropout_prob,
            run_tag=f'{candidate.tag}__selected',
            fold_index=fold_index,
            args=args,
            env=env,
        )
    evaluate_model(
        paths=final_paths,
        model_name='CECGaze',
        folds=folds,
        env=env,
        args=args,
        rerun_existing=True,
    )


def summarize_candidate(
    paths: StudyPaths,
    folds: list[int],
    candidate: CandidateConfig,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    summary_df = summarize_models(
        paths=paths,
        folds=folds,
        model_to_result_file={'CECGaze': 'trial_level_test_results.csv'},
    )[1]
    threshold_summary_df = summarize_thresholds(
        paths=paths,
        folds=folds,
        model_to_result_file={'CECGaze': 'trial_level_test_results.csv'},
    )

    val_average = summary_df[
        (summary_df['model'] == 'CECGaze')
        & (summary_df['eval_type'] == 'val')
        & (summary_df['eval_regime'] == 'average')
    ].iloc[0]
    test_average = summary_df[
        (summary_df['model'] == 'CECGaze')
        & (summary_df['eval_type'] == 'test')
        & (summary_df['eval_regime'] == 'average')
    ].iloc[0]
    threshold_test_average = threshold_summary_df[
        (threshold_summary_df['model'] == 'CECGaze')
        & (threshold_summary_df['eval_type'] == 'test')
        & (threshold_summary_df['eval_regime'] == 'average')
    ].iloc[0]

    metrics = {
        'candidate_tag': candidate.tag,
        'candidate_display_name': candidate.display_name,
        'learning_rate': candidate.learning_rate,
        'freeze_backbone': candidate.freeze_backbone,
        'dropout_prob': candidate.dropout_prob,
        'lambda_gold': float(args.lambda_gold),
        'lambda_annotator': float(args.lambda_annotator),
        'lambda_sparse': float(args.lambda_sparse),
        'val_average_auroc': float(val_average['auroc_mean']),
        'val_average_balanced_accuracy': float(val_average['balanced_accuracy_mean']),
        'test_average_auroc': float(test_average['auroc_mean']),
        'test_average_balanced_accuracy': float(test_average['balanced_accuracy_mean']),
        'test_average_balanced_accuracy_val_tuned': float(
            threshold_test_average['balanced_accuracy_val_tuned_mean']
        ),
        'selected_threshold_mean': float(threshold_test_average['selected_threshold_mean']),
    }
    metrics_path = paths.summary_root / 'candidate_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    return summary_df, threshold_summary_df, metrics


def load_candidate_metrics(candidate_root_path: Path) -> dict[str, float]:
    metrics_path = candidate_root_path / 'summary' / 'candidate_metrics.json'
    return json.loads(metrics_path.read_text())


def is_better_candidate(
    candidate_metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
) -> bool:
    if best_metrics is None:
        return True
    current_key = (
        float(candidate_metrics['val_average_auroc']),
        float(candidate_metrics['val_average_balanced_accuracy']),
    )
    best_key = (
        float(best_metrics['val_average_auroc']),
        float(best_metrics['val_average_balanced_accuracy']),
    )
    if current_key != best_key:
        return current_key > best_key
    return str(candidate_metrics['candidate_tag']) < str(best_metrics['candidate_tag'])


def materialize_best_candidate(
    output_root: Path,
    best_candidate_root: Path,
) -> None:
    canonical_model_root = output_root / 'CECGaze'
    source_root = best_candidate_root / 'CECGaze'
    if canonical_model_root.exists():
        shutil.rmtree(canonical_model_root)
    shutil.move(str(source_root), str(canonical_model_root))


def load_best_candidate(output_root: Path) -> CandidateConfig:
    best_config_path = output_root / 'summary' / 'best_config.json'
    if not best_config_path.exists():
        raise FileNotFoundError(
            f'Expected best-config metadata at {best_config_path}, but it is missing.'
        )
    payload = json.loads(best_config_path.read_text())
    return CandidateConfig(
        learning_rate=float(payload['learning_rate']),
        freeze_backbone=bool(payload['freeze_backbone']),
        dropout_prob=float(payload['dropout_prob']),
    )


def materialize_ablation_models(
    output_root: Path,
    candidate: CandidateConfig,
    folds: list[int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    final_paths = ensure_paths(output_root=output_root)
    for model_name in ABLATION_MODELS:
        logger.info(
            'Materializing ablation {} with best CEC config {}',
            model_name,
            candidate.display_name,
        )
        model_root = output_root / model_name
        for fold_index in folds:
            train_model_fold(
                paths=final_paths,
                model_name=model_name,
                learning_rate=candidate.learning_rate,
                freeze_backbone=candidate.freeze_backbone,
                dropout_prob=candidate.dropout_prob,
                run_tag=f'{candidate.tag}__{model_name}',
                fold_index=fold_index,
                args=args,
                env=env,
            )
            evaluate_model(
                paths=final_paths,
                model_name=model_name,
                folds=[fold_index],
                env=env,
                args=args,
                rerun_existing=args.rerun_existing,
            )
            remove_fold_checkpoints(model_root=model_root, fold_index=fold_index)


def build_markdown_report(
    output_root: Path,
    best_metrics: dict[str, float],
    leaderboard_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    score_drop_df: pd.DataFrame,
) -> None:
    report_path = output_root / 'summary' / 'cec_gaze_parity_sweep_report.md'
    reference_df = pd.concat(
        [load_reference_baselines('val'), load_reference_baselines('test')],
        ignore_index=True,
    )
    published_roberta_rows = reference_df[
        reference_df['model'] == 'Text-Only Roberta (official)'
    ][['model', 'eval_type', 'auroc_display', 'balanced_accuracy_display']].copy()

    tuned_rows = summary_df[
        (summary_df['eval_regime'] == 'average')
        & (summary_df['model'].isin(['CECGaze', 'CECGaze:uniform', 'CECGaze:shuffle']))
    ][['model', 'eval_type', 'auroc_display', 'balanced_accuracy_display']].copy()

    tuned_threshold_rows = threshold_summary_df[
        (threshold_summary_df['eval_type'] == 'test')
        & (threshold_summary_df['eval_regime'] == 'average')
    ][
        [
            'model',
            'selected_threshold_display',
            'balanced_accuracy_fixed_0_5_display',
            'balanced_accuracy_val_tuned_display',
        ]
    ].copy()

    leaderboard_view = leaderboard_df[
        [
            'rank',
            'candidate_tag',
            'learning_rate',
            'freeze_backbone',
            'dropout_prob',
            'val_average_auroc_display',
            'test_average_auroc_display',
            'test_average_balanced_accuracy_val_tuned_display',
        ]
    ].copy()

    score_drop_view = score_drop_df[
        [
            'eval_type',
            'eval_regime',
            'mean_abs_delta_top_drop_mean',
            'mean_abs_delta_random_drop_mean',
            'top_vs_random_abs_delta_ratio_mean',
            'auroc_delta_top_drop_mean',
            'auroc_delta_random_drop_mean',
        ]
    ].copy()

    lines = [
        '# CEC-Gaze Parity Sweep Report',
        '',
        '## Protocol',
        '',
        '- Published Text-only RoBERTa benchmark rows are shown for context only.',
        '- CEC-Gaze is tuned under the same style of validation-based model selection used for the benchmark family.',
        '- The sweep mirrors the benchmark tuning axes: learning rate, encoder freeze, and dropout.',
        '- Candidate ranking uses validation metrics only; test metrics are reported after selection.',
        (
            '- Auxiliary loss weights are fixed to '
            f'`lambda_gold={best_metrics.get("lambda_gold", 0.0)}`, '
            f'`lambda_annotator={best_metrics.get("lambda_annotator", 0.0)}`, and '
            f'`lambda_sparse={best_metrics.get("lambda_sparse", 0.0)}`.'
        ),
        '- Only the winning CEC candidate keeps fold checkpoints in the final output root.',
        '',
        '## Winning Config',
        '',
        f"- candidate: `{best_metrics['candidate_tag']}`",
        f"- learning_rate: `{best_metrics['learning_rate']}`",
        f"- freeze_backbone: `{best_metrics['freeze_backbone']}`",
        f"- dropout_prob: `{best_metrics['dropout_prob']}`",
        f"- lambda_gold: `{best_metrics.get('lambda_gold', 0.0)}`",
        f"- lambda_annotator: `{best_metrics.get('lambda_annotator', 0.0)}`",
        f"- lambda_sparse: `{best_metrics.get('lambda_sparse', 0.0)}`",
        f"- val average AUROC: `{100 * best_metrics['val_average_auroc']:.1f}`",
        f"- test average AUROC: `{100 * best_metrics['test_average_auroc']:.1f}`",
        '',
        '## Published Text-Only RoBERTa Context',
        '',
        safe_to_markdown(published_roberta_rows, index=False),
        '',
        '## Tuned CEC Metrics',
        '',
        safe_to_markdown(tuned_rows, index=False),
        '',
        '## Threshold-Tuned Balanced Accuracy',
        '',
        safe_to_markdown(tuned_threshold_rows, index=False),
        '',
        '## Sweep Leaderboard',
        '',
        safe_to_markdown(leaderboard_view, index=False),
        '',
        '## Score-Drop Summary',
        '',
        safe_to_markdown(score_drop_view, index=False, floatfmt='.4f'),
        '',
    ]
    report_path.write_text('\n'.join(lines) + '\n')
    logger.info('Saved report to {}', report_path)


def main() -> None:
    args = parse_args()
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f'Python interpreter not found at {PYTHON_BIN}')

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_root_path = sweep_root(
        output_root=output_root,
        requested_sweep_root=args.sweep_root.resolve() if args.sweep_root is not None else None,
    )
    sweep_root_path.mkdir(parents=True, exist_ok=True)

    have_final_main = final_outputs_exist(
        output_root=output_root,
        folds=args.folds,
        args=args,
    )
    have_final_ablations = ablation_outputs_exist(output_root=output_root, folds=args.folds)
    if have_final_main and have_final_ablations and not args.rerun_existing:
        logger.info('Selected CEC sweep outputs already exist under {}. Skipping.', output_root)
        return

    env = build_env()
    best_metrics: dict[str, float] | None = None
    best_candidate_root: Path | None = None
    leaderboard_rows: list[dict[str, object]] = []

    if args.rerun_existing or not have_final_main:
        configs = candidate_grid(args)
        logger.info(
            'Starting CEC-Gaze parity sweep with {} candidates over folds {}',
            len(configs),
            args.folds,
        )

        for candidate in configs:
            current_candidate_root = candidate_root(
                base_sweep_root=sweep_root_path,
                candidate=candidate,
            )
            current_paths = ensure_paths(output_root=current_candidate_root)

            logger.info('Evaluating candidate {}', candidate.display_name)
            if args.rerun_existing or not candidate_outputs_exist(
                candidate_root_path=current_candidate_root,
                args=args,
            ):
                if args.rerun_existing and current_candidate_root.exists():
                    shutil.rmtree(current_candidate_root)
                    current_paths = ensure_paths(output_root=current_candidate_root)
                for fold_index in args.folds:
                    train_candidate_fold(
                        paths=current_paths,
                        candidate=candidate,
                        fold_index=fold_index,
                        args=args,
                        env=env,
                    )
                    evaluate_model(
                        paths=current_paths,
                        model_name='CECGaze',
                        folds=[fold_index],
                        env=env,
                        args=args,
                        rerun_existing=args.rerun_existing,
                    )
                    remove_fold_checkpoints(
                        model_root=current_candidate_root / 'CECGaze',
                        fold_index=fold_index,
                    )
                summarize_candidate(
                    paths=current_paths,
                    folds=args.folds,
                    candidate=candidate,
                    args=args,
                )

            current_metrics = load_candidate_metrics(current_candidate_root)
            leaderboard_rows.append(current_metrics)

            if is_better_candidate(current_metrics, best_metrics):
                previous_best_root = best_candidate_root
                best_metrics = current_metrics
                best_candidate_root = current_candidate_root
                if (
                    previous_best_root is not None
                    and previous_best_root != best_candidate_root
                    and not args.keep_all_candidate_artifacts
                ):
                    prune_candidate_artifacts(previous_best_root)
            elif not args.keep_all_candidate_artifacts:
                prune_candidate_artifacts(current_candidate_root)

        if best_metrics is None or best_candidate_root is None:
            raise RuntimeError('No CEC sweep candidates completed successfully.')

        materialize_best_candidate(
            output_root=output_root,
            best_candidate_root=best_candidate_root,
        )
        selected_candidate = CandidateConfig(
            learning_rate=float(best_metrics['learning_rate']),
            freeze_backbone=bool(best_metrics['freeze_backbone']),
            dropout_prob=float(best_metrics['dropout_prob']),
        )
    else:
        logger.info(
            'Selected CEC main outputs already exist under {}. Reusing the stored best config.',
            output_root,
        )
        selected_candidate = load_best_candidate(output_root=output_root)

    final_paths = ensure_paths(output_root=output_root)
    materialize_main_model(
        output_root=output_root,
        candidate=selected_candidate,
        folds=args.folds,
        args=args,
        env=env,
    )
    materialize_ablation_models(
        output_root=output_root,
        candidate=selected_candidate,
        folds=args.folds,
        args=args,
        env=env,
    )
    evaluate_model(
        paths=final_paths,
        model_name='CECGaze',
        folds=args.folds,
        env=env,
        args=args,
        rerun_existing=False,
        score_eval_mode='uniform',
    )
    evaluate_model(
        paths=final_paths,
        model_name='CECGaze',
        folds=args.folds,
        env=env,
        args=args,
        rerun_existing=False,
        score_eval_mode='shuffle',
    )
    run_score_drop(paths=final_paths, env=env, args=args)

    if not args.keep_final_wandb_offline_runs:
        for model_name in ['CECGaze', *ABLATION_MODELS]:
            remove_wandb_runs(output_root / model_name)

    model_to_result_file = {
        'CECGaze': 'trial_level_test_results.csv',
        'CECGaze:uniform': 'trial_level_test_results_uniform.csv',
        'CECGaze:shuffle': 'trial_level_test_results_shuffle.csv',
    }
    _, summary_df = summarize_models(
        paths=final_paths,
        folds=args.folds,
        model_to_result_file=model_to_result_file,
    )
    threshold_summary_df = summarize_thresholds(
        paths=final_paths,
        folds=args.folds,
        model_to_result_file=model_to_result_file,
    )
    score_drop_df = summarize_score_drop(paths=final_paths, folds=args.folds)

    if leaderboard_rows:
        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
            ['val_average_auroc', 'val_average_balanced_accuracy', 'candidate_tag'],
            ascending=[False, False, True],
        )
        leaderboard_df.insert(0, 'rank', range(1, len(leaderboard_df) + 1))
        for metric_name in [
            'val_average_auroc',
            'test_average_auroc',
            'val_average_balanced_accuracy',
            'test_average_balanced_accuracy',
            'test_average_balanced_accuracy_val_tuned',
        ]:
            leaderboard_df[f'{metric_name}_display'] = leaderboard_df[metric_name].map(
                lambda value: f'{100 * float(value):.1f}'
            )
        leaderboard_df.to_csv(
            output_root / 'summary' / 'cec_gaze_sweep_leaderboard.csv',
            index=False,
        )

        best_config_payload = dict(best_metrics)
        best_config_payload['sweep_size'] = len(configs)
        best_config_payload['folds'] = args.folds
        best_config_payload['backbone'] = args.backbone
        best_config_payload['batch_size'] = args.batch_size
        best_config_payload['max_epochs'] = args.max_epochs
        best_config_payload['selection_protocol'] = 'validation_only'
        (output_root / 'summary' / 'best_config.json').write_text(
            json.dumps(best_config_payload, indent=2, sort_keys=True) + '\n'
        )
    else:
        leaderboard_path = output_root / 'summary' / 'cec_gaze_sweep_leaderboard.csv'
        if leaderboard_path.exists():
            leaderboard_df = pd.read_csv(leaderboard_path)
        else:
            leaderboard_df = pd.DataFrame(
                [
                    {
                        'rank': 1,
                        'candidate_tag': selected_candidate.tag,
                        'learning_rate': selected_candidate.learning_rate,
                        'freeze_backbone': selected_candidate.freeze_backbone,
                        'dropout_prob': selected_candidate.dropout_prob,
                        'val_average_auroc_display': 'n/a',
                        'test_average_auroc_display': 'n/a',
                        'test_average_balanced_accuracy_val_tuned_display': 'n/a',
                    }
                ]
            )
        best_metrics = load_candidate_metrics(
            candidate_root(
                base_sweep_root=sweep_root_path,
                candidate=selected_candidate,
            )
        ) if candidate_outputs_exist(
            candidate_root(
                base_sweep_root=sweep_root_path,
                candidate=selected_candidate,
            ),
            args=args,
        ) else {
            'candidate_tag': selected_candidate.tag,
            'learning_rate': selected_candidate.learning_rate,
            'freeze_backbone': selected_candidate.freeze_backbone,
            'dropout_prob': selected_candidate.dropout_prob,
            'lambda_gold': args.lambda_gold,
            'lambda_annotator': args.lambda_annotator,
            'lambda_sparse': args.lambda_sparse,
            'val_average_auroc': float('nan'),
            'test_average_auroc': float('nan'),
        }

    build_markdown_report(
        output_root=output_root,
        best_metrics=best_metrics,
        leaderboard_df=leaderboard_df,
        summary_df=summary_df,
        threshold_summary_df=threshold_summary_df,
        score_drop_df=score_drop_df,
    )
    logger.info(
        'Finished CEC-Gaze parity sweep. Best candidate {} selected under {}',
        best_metrics['candidate_tag'],
        output_root / 'CECGaze',
    )


if __name__ == '__main__':
    main()
