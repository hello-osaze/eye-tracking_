from __future__ import annotations

import argparse
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = REPO_ROOT.parent / '.venv' / 'bin' / 'python'

LOCAL_MODELS = [
    'CECGaze',
    'Roberta',
    'RoberteyeWord',
    'RoberteyeFixation',
    'MAG',
    'PostFusion',
]
REFERENCE_BASELINES = [
    'Text-Only Roberta',
    'RoBERTEye-W~\\citep{Shubi2024Finegrained}',
    'RoBERTEye-F~\\citep{Shubi2024Finegrained}',
    'MAG-Eye~\\citep{Shubi2024Finegrained}',
    'PostFusion-Eye~\\citep{Shubi2024Finegrained}',
]
LOCAL_DISPLAY_NAMES = {
    'CECGaze': 'CECGaze (local)',
    'Roberta': 'Text-Only Roberta (local)',
    'RoberteyeWord': 'RoBERTEye-W (local)',
    'RoberteyeFixation': 'RoBERTEye-F (local)',
    'MAG': 'MAG-Eye (local)',
    'PostFusion': 'PostFusion-Eye (local)',
}


@dataclass
class StudyPaths:
    output_root: Path
    summary_root: Path
    logs_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a local IITBHGC benchmark suite under one shared budget.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('outputs/iitbhgc_local_benchmark_suite'),
        help='Directory where fold checkpoints and evaluation CSVs are stored.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=LOCAL_MODELS,
        help='Registered Hydra model config names to train and evaluate.',
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
        default=2,
        help='Per-step batch size override.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=10,
        help='Number of epochs per fold for this local benchmark run.',
    )
    parser.add_argument(
        '--max-time-limit',
        default=None,
        help=(
            'Optional Lightning max_time override per fold, formatted as DD:HH:MM:SS. '
            'When omitted, the model default sweep-derived time budget is used.'
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
        '--wandb-project',
        default='IITBHGCLocalBenchmarkSuite',
        help='WandB project name. Runs are forced offline by this script.',
    )
    parser.add_argument(
        '--unfreeze-backbone',
        action='store_true',
        help='Fine-tune the text encoder instead of freezing it.',
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Retrain/evaluate even if expected artifacts already exist.',
    )
    return parser.parse_args()


def ensure_paths(output_root: Path) -> StudyPaths:
    summary_root = output_root / 'summary'
    logs_root = output_root / 'logs'
    summary_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    return StudyPaths(
        output_root=output_root,
        summary_root=summary_root,
        logs_root=logs_root,
    )


def tail_log(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ''
    lines = path.read_text(errors='replace').splitlines()
    return '\n'.join(lines[-max_lines:])


def run_command(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Running: {}', ' '.join(cmd))
    with log_path.open('a') as log_file:
        log_file.write(f'\n\n===== COMMAND =====\n{" ".join(cmd)}\n\n')
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f'Command failed with exit code {result.returncode}: {" ".join(cmd)}\n'
            f'Log tail from {log_path}:\n{tail_log(log_path)}'
        )


def fold_dir(output_root: Path, model_name: str, fold_index: int) -> Path:
    return output_root / model_name / f'fold_index={fold_index}'


def has_checkpoint(path: Path) -> bool:
    return any(path.glob('*lowest_loss_val_all*.ckpt'))


def all_eval_outputs_exist(
    output_root: Path,
    model_name: str,
    folds: list[int],
    result_filename: str,
) -> bool:
    return all(
        (fold_dir(output_root, model_name, fold_index) / result_filename).exists()
        for fold_index in folds
    )


def freeze_override(model_name: str, unfreeze_backbone: bool) -> str:
    freeze_value = str(not unfreeze_backbone).lower()
    if model_name.startswith('CECGaze'):
        return f'model.freeze_backbone={freeze_value}'
    return f'model.freeze={freeze_value}'


def train_fold(
    paths: StudyPaths,
    model_name: str,
    fold_index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    current_fold_dir = fold_dir(paths.output_root, model_name, fold_index)
    if has_checkpoint(current_fold_dir) and not args.rerun_existing:
        logger.info(
            'Skipping train {} fold {}: checkpoint already exists',
            model_name,
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
        'trainer.num_workers=0',
        f'trainer.wandb_project={args.wandb_project}',
        f'trainer.wandb_job_type={model_name}_fold{fold_index}',
        f'model.backbone={args.backbone}',
        f'model.batch_size={args.batch_size}',
        'model.accumulate_grad_batches=1',
        f'model.max_epochs={args.max_epochs}',
        *(
            [f'model.max_time_limit={args.max_time_limit}']
            if args.max_time_limit is not None
            else []
        ),
        freeze_override(
            model_name=model_name,
            unfreeze_backbone=args.unfreeze_backbone,
        ),
        f'hydra.run.dir={hydra_run_dir}',
    ]
    run_command(
        cmd=cmd,
        log_path=paths.logs_root / f'train_{model_name}_fold_{fold_index}.log',
        env=env,
    )


def evaluate_model(
    paths: StudyPaths,
    model_name: str,
    folds: list[int],
    env: dict[str, str],
    rerun_existing: bool,
) -> str:
    result_filename = 'trial_level_test_results.csv'
    if (
        all_eval_outputs_exist(paths.output_root, model_name, folds, result_filename)
        and not rerun_existing
    ):
        logger.info('Skipping eval {}: prediction CSVs already exist', model_name)
        return result_filename

    cmd = [
        str(PYTHON_BIN),
        'src/run/single_run/test_dl.py',
        f'eval_path={paths.output_root / model_name}',
    ]
    run_command(
        cmd=cmd,
        log_path=paths.logs_root / f'eval_{model_name}.log',
        env=env,
    )
    return result_filename


def safe_auroc(labels: pd.Series, prediction_prob: pd.Series) -> float:
    if labels.nunique() < 2:
        return float('nan')
    return float(roc_auc_score(labels, prediction_prob))


def safe_balanced_accuracy(
    labels: pd.Series,
    prediction_prob: pd.Series,
    threshold: float,
) -> float:
    return float(
        balanced_accuracy_score(labels, (prediction_prob >= threshold).astype(int))
    )


def select_balanced_accuracy_threshold(
    labels: pd.Series,
    prediction_prob: pd.Series,
) -> float:
    if labels.nunique() < 2 or prediction_prob.nunique() < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels, prediction_prob)
    finite_mask = pd.notna(thresholds)
    if not finite_mask.any():
        return 0.5

    finite_fpr = fpr[finite_mask]
    finite_tpr = tpr[finite_mask]
    finite_thresholds = thresholds[finite_mask]
    j_scores = finite_tpr - finite_fpr
    best_index = int(j_scores.argmax())
    return float(finite_thresholds[best_index])


def compute_fold_metrics(
    model_name: str,
    trial_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (eval_type, fold_index, eval_regime), group_df in trial_results.groupby(
        ['eval_type', 'fold_index', 'eval_regime']
    ):
        labels = group_df['label']
        probs = group_df['prediction_prob']
        rows.append(
            {
                'model': model_name,
                'eval_type': eval_type,
                'fold_index': fold_index,
                'eval_regime': eval_regime,
                'auroc': safe_auroc(labels, probs),
                'balanced_accuracy': safe_balanced_accuracy(
                    labels=labels,
                    prediction_prob=probs,
                    threshold=0.5,
                ),
                'n_samples': len(group_df),
            }
        )

    metrics_df = pd.DataFrame(rows)
    average_rows = []
    for (eval_type, fold_index), group_df in metrics_df.groupby(
        ['eval_type', 'fold_index']
    ):
        average_rows.append(
            {
                'model': model_name,
                'eval_type': eval_type,
                'fold_index': fold_index,
                'eval_regime': 'average',
                'auroc': group_df['auroc'].mean(),
                'balanced_accuracy': group_df['balanced_accuracy'].mean(),
                'n_samples': group_df['n_samples'].sum(),
            }
        )
    return pd.concat([metrics_df, pd.DataFrame(average_rows)], ignore_index=True)


def aggregate_fold_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (
        model_name,
        eval_type,
        eval_regime,
    ), group_df in fold_metrics_df.groupby(['model', 'eval_type', 'eval_regime']):
        row = {
            'model': model_name,
            'eval_type': eval_type,
            'eval_regime': eval_regime,
            'n_folds': group_df['fold_index'].nunique(),
        }
        for metric_name in ['auroc', 'balanced_accuracy']:
            values = group_df[metric_name].dropna()
            mean_value = float(values.mean()) if len(values) else float('nan')
            sem_value = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            row[f'{metric_name}_mean'] = mean_value
            row[f'{metric_name}_sem'] = sem_value
            row[f'{metric_name}_display'] = (
                'nan'
                if math.isnan(mean_value)
                else f'{100 * mean_value:.1f} ± {100 * sem_value:.1f}'
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['eval_type', 'model', 'eval_regime'])


def load_trial_results(
    model_dir: Path,
    folds: list[int],
    result_filename: str,
) -> pd.DataFrame:
    result_frames = []
    for fold_index in folds:
        result_path = model_dir / f'fold_index={fold_index}' / result_filename
        result_frames.append(pd.read_csv(result_path))
    return pd.concat(result_frames, ignore_index=True)


def summarize_models(
    paths: StudyPaths,
    folds: list[int],
    model_to_result_file: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_metric_frames = []
    for model_name, result_filename in model_to_result_file.items():
        trial_results = load_trial_results(
            model_dir=paths.output_root / model_name,
            folds=folds,
            result_filename=result_filename,
        )
        fold_metric_frames.append(
            compute_fold_metrics(model_name=model_name, trial_results=trial_results)
        )

    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    summary_df = aggregate_fold_metrics(fold_metrics_df)
    fold_metrics_df.to_csv(
        paths.summary_root / 'local_benchmark_fold_metrics.csv',
        index=False,
    )
    summary_df.to_csv(
        paths.summary_root / 'local_benchmark_summary_metrics.csv',
        index=False,
    )
    return fold_metrics_df, summary_df


def compute_threshold_fold_metrics(
    model_name: str,
    trial_results: pd.DataFrame,
) -> pd.DataFrame:
    threshold_rows = []
    for (fold_index, eval_regime), val_df in trial_results[
        trial_results['eval_type'] == 'val'
    ].groupby(['fold_index', 'eval_regime']):
        selected_threshold = select_balanced_accuracy_threshold(
            labels=val_df['label'],
            prediction_prob=val_df['prediction_prob'],
        )
        threshold_rows.append(
            {
                'fold_index': fold_index,
                'eval_regime': eval_regime,
                'selected_threshold': selected_threshold,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    rows = []
    for (eval_type, fold_index, eval_regime), group_df in trial_results.groupby(
        ['eval_type', 'fold_index', 'eval_regime']
    ):
        selected_threshold = float(
            threshold_df[
                (threshold_df['fold_index'] == fold_index)
                & (threshold_df['eval_regime'] == eval_regime)
            ]['selected_threshold'].iloc[0]
        )
        labels = group_df['label']
        probs = group_df['prediction_prob']
        rows.append(
            {
                'model': model_name,
                'eval_type': eval_type,
                'fold_index': fold_index,
                'eval_regime': eval_regime,
                'selected_threshold': selected_threshold,
                'balanced_accuracy_fixed_0_5': safe_balanced_accuracy(
                    labels=labels,
                    prediction_prob=probs,
                    threshold=0.5,
                ),
                'balanced_accuracy_val_tuned': safe_balanced_accuracy(
                    labels=labels,
                    prediction_prob=probs,
                    threshold=selected_threshold,
                ),
                'n_samples': len(group_df),
            }
        )

    fold_metrics_df = pd.DataFrame(rows)
    average_rows = []
    for (eval_type, fold_index), group_df in fold_metrics_df.groupby(
        ['eval_type', 'fold_index']
    ):
        average_rows.append(
            {
                'model': model_name,
                'eval_type': eval_type,
                'fold_index': fold_index,
                'eval_regime': 'average',
                'selected_threshold': group_df['selected_threshold'].mean(),
                'balanced_accuracy_fixed_0_5': group_df[
                    'balanced_accuracy_fixed_0_5'
                ].mean(),
                'balanced_accuracy_val_tuned': group_df[
                    'balanced_accuracy_val_tuned'
                ].mean(),
                'n_samples': group_df['n_samples'].sum(),
            }
        )
    return pd.concat([fold_metrics_df, pd.DataFrame(average_rows)], ignore_index=True)


def aggregate_threshold_metrics(
    threshold_fold_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        model_name,
        eval_type,
        eval_regime,
    ), group_df in threshold_fold_metrics_df.groupby(
        ['model', 'eval_type', 'eval_regime']
    ):
        row = {
            'model': model_name,
            'eval_type': eval_type,
            'eval_regime': eval_regime,
            'n_folds': group_df['fold_index'].nunique(),
        }
        for metric_name in [
            'selected_threshold',
            'balanced_accuracy_fixed_0_5',
            'balanced_accuracy_val_tuned',
        ]:
            values = group_df[metric_name].dropna()
            mean_value = float(values.mean()) if len(values) else float('nan')
            sem_value = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            row[f'{metric_name}_mean'] = mean_value
            row[f'{metric_name}_sem'] = sem_value
            if metric_name.startswith('balanced_accuracy'):
                row[f'{metric_name}_display'] = (
                    'nan'
                    if math.isnan(mean_value)
                    else f'{100 * mean_value:.1f} ± {100 * sem_value:.1f}'
                )
            else:
                row[f'{metric_name}_display'] = (
                    'nan'
                    if math.isnan(mean_value)
                    else f'{mean_value:.3f} ± {sem_value:.3f}'
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['eval_type', 'model', 'eval_regime'])


def summarize_thresholds(
    paths: StudyPaths,
    folds: list[int],
    model_to_result_file: dict[str, str],
) -> pd.DataFrame:
    threshold_fold_frames = []
    for model_name, result_filename in model_to_result_file.items():
        trial_results = load_trial_results(
            model_dir=paths.output_root / model_name,
            folds=folds,
            result_filename=result_filename,
        )
        threshold_fold_frames.append(
            compute_threshold_fold_metrics(
                model_name=model_name,
                trial_results=trial_results,
            )
        )

    threshold_fold_metrics_df = pd.concat(threshold_fold_frames, ignore_index=True)
    threshold_summary_df = aggregate_threshold_metrics(
        threshold_fold_metrics_df=threshold_fold_metrics_df
    )
    threshold_fold_metrics_df.to_csv(
        paths.summary_root / 'local_benchmark_threshold_fold_metrics.csv',
        index=False,
    )
    threshold_summary_df.to_csv(
        paths.summary_root / 'local_benchmark_threshold_summary_metrics.csv',
        index=False,
    )
    return threshold_summary_df


def parse_pm_value(value: str) -> tuple[float, float]:
    mean_text, sem_text = value.split('±')
    return float(mean_text.strip()), float(sem_text.strip())


def load_reference_baselines(eval_type: str) -> pd.DataFrame:
    csv_path = (
        REPO_ROOT
        / 'results'
        / 'formatted_eyebench_benchmark_results'
        / f'IITBHGC_CV_{eval_type}.csv'
    )
    baseline_df = pd.read_csv(csv_path)
    baseline_df = baseline_df[baseline_df['Model'].isin(REFERENCE_BASELINES)].copy()
    rename_map = {
        'Text-Only Roberta': 'Text-Only Roberta (official)',
        'RoBERTEye-W~\\citep{Shubi2024Finegrained}': 'RoBERTEye-W (official)',
        'RoBERTEye-F~\\citep{Shubi2024Finegrained}': 'RoBERTEye-F (official)',
        'MAG-Eye~\\citep{Shubi2024Finegrained}': 'MAG-Eye (official)',
        'PostFusion-Eye~\\citep{Shubi2024Finegrained}': 'PostFusion-Eye (official)',
    }
    rows = []
    for _, row in baseline_df.iterrows():
        auroc_mean, auroc_sem = parse_pm_value(row['Average_AUROC'])
        bacc_mean, bacc_sem = parse_pm_value(
            row[r'Average_\makecell{Balanced\\Accuracy}']
        )
        rows.append(
            {
                'model': rename_map[row['Model']],
                'eval_type': eval_type,
                'eval_regime': 'average',
                'n_folds': 4,
                'auroc_mean': auroc_mean / 100.0,
                'auroc_sem': auroc_sem / 100.0,
                'auroc_display': f'{auroc_mean:.1f} ± {auroc_sem:.1f}',
                'balanced_accuracy_mean': bacc_mean / 100.0,
                'balanced_accuracy_sem': bacc_sem / 100.0,
                'balanced_accuracy_display': f'{bacc_mean:.1f} ± {bacc_sem:.1f}',
            }
        )
    return pd.DataFrame(rows)


def display_model_name(model_name: str) -> str:
    return LOCAL_DISPLAY_NAMES.get(model_name, model_name)


def build_markdown_report(
    paths: StudyPaths,
    summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    report_path = paths.summary_root / 'local_benchmark_report.md'

    local_rows = summary_df.copy()
    local_rows['model_display'] = local_rows['model'].map(display_model_name)

    combined_sections = []
    for eval_type in ['val', 'test']:
        local_table = local_rows[
            (local_rows['eval_type'] == eval_type)
            & (local_rows['eval_regime'] == 'average')
        ][
            ['model_display', 'auroc_display', 'balanced_accuracy_display']
        ].rename(columns={'model_display': 'model'})

        ref_table = reference_df[reference_df['eval_type'] == eval_type][
            ['model', 'auroc_display', 'balanced_accuracy_display']
        ]

        combined_sections.append(
            f'## {eval_type.upper()} Local Same-Budget Metrics\n\n'
            + local_table.to_markdown(index=False)
        )
        combined_sections.append(
            f'## {eval_type.upper()} Official Reference Metrics\n\n'
            + ref_table.to_markdown(index=False)
        )

    threshold_table = threshold_summary_df[
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
    threshold_table['model'] = threshold_table['model'].map(display_model_name)

    test_average = summary_df[
        (summary_df['eval_type'] == 'test') & (summary_df['eval_regime'] == 'average')
    ].set_index('model')
    best_local_model = test_average['auroc_mean'].idxmax()
    best_local_auroc = test_average.loc[best_local_model, 'auroc_mean']
    interpretation_lines = [
        f'- Best local same-budget model: {display_model_name(best_local_model)} ({best_local_auroc:.4f})',
    ]
    if {'CECGaze', 'Roberta'}.issubset(test_average.index):
        cec_auroc = test_average.loc['CECGaze', 'auroc_mean']
        local_roberta = test_average.loc['Roberta', 'auroc_mean']
        interpretation_lines.insert(
            0,
            f'- Test Avg AUROC CECGaze vs local Text-Only Roberta: {cec_auroc:.4f} / {local_roberta:.4f}',
        )
    elif 'CECGaze' in test_average.index:
        cec_auroc = test_average.loc['CECGaze', 'auroc_mean']
        interpretation_lines.insert(
            0,
            f'- Test Avg AUROC CECGaze: {cec_auroc:.4f}',
        )

    official_test_average = reference_df[
        (reference_df['eval_type'] == 'test')
        & (reference_df['eval_regime'] == 'average')
    ].set_index('model')
    best_official_model = official_test_average['auroc_mean'].idxmax()
    best_official_auroc = official_test_average.loc[best_official_model, 'auroc_mean']

    lines = [
        '# IITBHGC Local Benchmark Report',
        '',
        '## Protocol',
        '',
        f'- models: `{args.models}`',
        f'- backbone: `{args.backbone}`',
        f'- max_epochs per fold: `{args.max_epochs}`',
        f'- batch_size: `{args.batch_size}`',
        f'- trainer accelerator/devices: `{args.trainer_accelerator}` / `{args.trainer_devices}`',
        f'- encoder frozen: `{not args.unfreeze_backbone}`',
        f'- folds: `{args.folds}`',
        '- All local models in this report are retrained under the same local budget and evaluated with the same test pipeline.',
        '- Official EyeBench rows are included only as published reference context.',
        '',
        *combined_sections,
        '',
        '## Threshold-Tuned Balanced Accuracy',
        '',
        '- `selected_threshold_display` is chosen on the matching validation fold/regime, then applied to the test split of that same fold/regime.',
        '- This is a calibration diagnostic; AUROC is unchanged by threshold choice.',
        '',
        threshold_table.to_markdown(index=False),
        '',
        '## Interpretation Checks',
        '',
        *interpretation_lines,
        f'- Best official reference model on test: {best_official_model} ({best_official_auroc:.4f})',
        '- If CECGaze exceeds the local text-only baseline but not the strongest local eye-aware baselines, that supports the claim that the architecture is viable but not yet dominant.',
        '- If all local models collapse to balanced accuracy 0.5 at threshold 0.5, interpret AUROC and val-tuned balanced accuracy as the more informative diagnostics.',
    ]

    report_path.write_text('\n'.join(lines) + '\n')
    logger.info('Saved report to {}', report_path)


def main() -> None:
    args = parse_args()
    paths = ensure_paths(output_root=args.output_root)
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['HYDRA_FULL_ERROR'] = '1'
    env['HF_HUB_DISABLE_XET'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['MPLCONFIGDIR'] = '/tmp/matplotlib-eyebench'

    logger.info(
        'Starting local IITBHGC benchmark suite: models={}, folds={}, backbone={}, max_epochs={}, accelerator={}',
        args.models,
        args.folds,
        args.backbone,
        args.max_epochs,
        args.trainer_accelerator,
    )

    for model_name in args.models:
        for fold_index in args.folds:
            train_fold(
                paths=paths,
                model_name=model_name,
                fold_index=fold_index,
                args=args,
                env=env,
            )
        evaluate_model(
            paths=paths,
            model_name=model_name,
            folds=args.folds,
            env=env,
            rerun_existing=args.rerun_existing,
        )

    model_to_result_file = {
        model_name: 'trial_level_test_results.csv' for model_name in args.models
    }
    _, summary_df = summarize_models(
        paths=paths,
        folds=args.folds,
        model_to_result_file=model_to_result_file,
    )
    threshold_summary_df = summarize_thresholds(
        paths=paths,
        folds=args.folds,
        model_to_result_file=model_to_result_file,
    )
    reference_df = pd.concat(
        [load_reference_baselines('val'), load_reference_baselines('test')],
        ignore_index=True,
    )
    reference_df.to_csv(
        paths.summary_root / 'official_baseline_reference_metrics.csv',
        index=False,
    )
    build_markdown_report(
        paths=paths,
        summary_df=summary_df,
        threshold_summary_df=threshold_summary_df,
        reference_df=reference_df,
        args=args,
    )
    logger.info('Finished local IITBHGC benchmark suite.')


if __name__ == '__main__':
    main()
