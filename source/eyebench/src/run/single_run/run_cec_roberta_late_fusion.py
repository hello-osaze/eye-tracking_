from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from src.run.single_run.report_tables import safe_to_markdown


REGIMES = [
    'seen_subject_unseen_item',
    'unseen_subject_seen_item',
    'unseen_subject_unseen_item',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Blend Text-only RoBERTa and CEC-Gaze trial predictions with a '
            'fold-local validation-tuned interpolation weight.'
        )
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=Path(
            'results/raw/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
            'trainer.wandb_job_type=Roberta_IITBHGC_CV'
        ),
        help='Directory containing RoBERTa fold_index=*/trial_level_test_results.csv files.',
    )
    parser.add_argument(
        '--cec-root',
        type=Path,
        default=Path('outputs/cec_gaze_claim_context_mlp_mps_large_true_e10/CECGaze'),
        help='Directory containing CEC fold_index=*/trial-level prediction files.',
    )
    parser.add_argument(
        '--cec-result-filename',
        default='trial_level_test_results.csv',
        help='CEC prediction filename inside each fold directory.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path(
            'outputs/cec_roberta_late_fusion_mps_large_true_e10/CECGazeRobertaValBlendFine'
        ),
        help='Where blended fold predictions and summary files are written.',
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        help='Fold indices to blend.',
    )
    parser.add_argument(
        '--alpha-grid-step',
        type=float,
        default=0.001,
        help='Grid step for CEC interpolation weight. alpha=0 is RoBERTa only.',
    )
    parser.add_argument(
        '--model-name',
        default='CECGaze+RoBERTaValBlendFine-true-e10',
        help='Model label used in summary CSVs.',
    )
    parser.add_argument(
        '--official-baseline-csv',
        type=Path,
        default=Path('results/formatted_eyebench_benchmark_results/IITBHGC_CV_test.csv'),
        help='Optional formatted benchmark CSV for report comparison.',
    )
    return parser.parse_args()


def read_predictions(root: Path, fold_index: int, filename: str) -> pd.DataFrame:
    path = root / f'fold_index={fold_index}' / filename
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    required_columns = {
        'label',
        'prediction_prob',
        'eval_regime',
        'eval_type',
        'fold_index',
        'unique_trial_id',
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f'{path} is missing columns: {missing_columns}')
    return df


def mean_regime_auroc(df: pd.DataFrame, score_column: str, eval_type: str) -> float:
    values = []
    for regime in REGIMES:
        regime_df = df[
            (df['eval_type'] == eval_type) & (df['eval_regime'] == regime)
        ]
        values.append(
            roc_auc_score(
                regime_df['label'].astype(int),
                regime_df[score_column].astype(float),
            )
        )
    return float(np.mean(values))


def alpha_grid(step: float) -> np.ndarray:
    if step <= 0 or step > 1:
        raise ValueError('--alpha-grid-step must be in (0, 1].')
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    return np.clip(grid, 0.0, 1.0)


def select_fold_alpha(fold_df: pd.DataFrame, step: float) -> tuple[float, float]:
    best_alpha = 0.0
    best_auroc = -math.inf
    for alpha in alpha_grid(step):
        fold_df['candidate_prob'] = (
            (1.0 - alpha) * fold_df['roberta_prediction_prob']
            + alpha * fold_df['cec_prediction_prob']
        )
        val_auroc = mean_regime_auroc(
            df=fold_df,
            score_column='candidate_prob',
            eval_type='val',
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_alpha = float(alpha)
    fold_df.drop(columns=['candidate_prob'], inplace=True)
    return best_alpha, best_auroc


def blend_fold(
    roberta_df: pd.DataFrame,
    cec_df: pd.DataFrame,
    alpha_step: float,
) -> tuple[pd.DataFrame, float, float]:
    merge_keys = [
        'fold_index',
        'eval_type',
        'eval_regime',
        'unique_trial_id',
    ]
    roberta_columns = [
        *merge_keys,
        'label',
        'prediction_prob',
        'unique_paragraph_id',
        'participant_id',
    ]
    roberta_columns = [column for column in roberta_columns if column in roberta_df.columns]
    cec_columns = [*merge_keys, 'label', 'prediction_prob']

    merged = roberta_df[roberta_columns].merge(
        cec_df[cec_columns],
        on=merge_keys,
        how='inner',
        suffixes=('_roberta', '_cec'),
    )
    if len(merged) != len(roberta_df) or len(merged) != len(cec_df):
        raise ValueError(
            'RoBERTa and CEC predictions do not align exactly: '
            f'roberta={len(roberta_df)}, cec={len(cec_df)}, merged={len(merged)}'
        )
    if not (merged['label_roberta'].astype(int) == merged['label_cec'].astype(int)).all():
        raise ValueError('RoBERTa and CEC labels disagree after merge.')

    merged = merged.rename(
        columns={
            'label_roberta': 'label',
            'prediction_prob_roberta': 'roberta_prediction_prob',
            'prediction_prob_cec': 'cec_prediction_prob',
        }
    )
    merged = merged.drop(columns=['label_cec'])

    alpha, val_auroc = select_fold_alpha(fold_df=merged, step=alpha_step)
    merged['prediction_prob'] = (
        (1.0 - alpha) * merged['roberta_prediction_prob']
        + alpha * merged['cec_prediction_prob']
    )
    merged['cec_blend_alpha'] = alpha

    preferred_columns = [
        'label',
        'prediction_prob',
        'eval_regime',
        'eval_type',
        'fold_index',
        'unique_paragraph_id',
        'participant_id',
        'unique_trial_id',
        'roberta_prediction_prob',
        'cec_prediction_prob',
        'cec_blend_alpha',
    ]
    ordered_columns = [column for column in preferred_columns if column in merged.columns]
    extra_columns = [column for column in merged.columns if column not in ordered_columns]
    return merged[ordered_columns + extra_columns].copy(), alpha, val_auroc


def compute_fold_metrics(model_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eval_type, fold_index, regime), group_df in predictions.groupby(
        ['eval_type', 'fold_index', 'eval_regime']
    ):
        labels = group_df['label'].astype(int)
        scores = group_df['prediction_prob'].astype(float)
        rows.append(
            {
                'model': model_name,
                'eval_type': eval_type,
                'fold_index': int(fold_index),
                'eval_regime': regime,
                'auroc': roc_auc_score(labels, scores),
                'balanced_accuracy': balanced_accuracy_score(
                    labels,
                    (scores >= 0.5).astype(int),
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
                'fold_index': int(fold_index),
                'eval_regime': 'average',
                'auroc': float(group_df['auroc'].mean()),
                'balanced_accuracy': float(group_df['balanced_accuracy'].mean()),
                'n_samples': int(group_df['n_samples'].sum()),
            }
        )
    return pd.concat([metrics_df, pd.DataFrame(average_rows)], ignore_index=True)


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, eval_type, regime), group_df in fold_metrics.groupby(
        ['model', 'eval_type', 'eval_regime']
    ):
        aurocs = group_df['auroc']
        baccs = group_df['balanced_accuracy']
        auroc_mean = float(aurocs.mean())
        bacc_mean = float(baccs.mean())
        auroc_sem = float(aurocs.std(ddof=1) / math.sqrt(len(aurocs))) if len(aurocs) > 1 else 0.0
        bacc_sem = float(baccs.std(ddof=1) / math.sqrt(len(baccs))) if len(baccs) > 1 else 0.0
        rows.append(
            {
                'model': model,
                'eval_type': eval_type,
                'eval_regime': regime,
                'n_folds': group_df['fold_index'].nunique(),
                'auroc_mean': auroc_mean,
                'auroc_sem': auroc_sem,
                'auroc_display': f'{100 * auroc_mean:.1f} +/- {100 * auroc_sem:.1f}',
                'balanced_accuracy_mean': bacc_mean,
                'balanced_accuracy_sem': bacc_sem,
                'balanced_accuracy_display': (
                    f'{100 * bacc_mean:.1f} +/- {100 * bacc_sem:.1f}'
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(['eval_type', 'model', 'eval_regime'])


def select_threshold(labels: pd.Series, scores: pd.Series) -> float:
    fpr, tpr, thresholds = roc_curve(labels.astype(int), scores.astype(float))
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    if math.isinf(threshold):
        threshold = float(scores.max()) + 1e-6
    return threshold


def compute_threshold_fold_metrics(
    model_name: str,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fold_index in sorted(predictions['fold_index'].unique()):
        for regime in REGIMES:
            val_df = predictions[
                (predictions['fold_index'] == fold_index)
                & (predictions['eval_type'] == 'val')
                & (predictions['eval_regime'] == regime)
            ]
            test_df = predictions[
                (predictions['fold_index'] == fold_index)
                & (predictions['eval_type'] == 'test')
                & (predictions['eval_regime'] == regime)
            ]
            threshold = select_threshold(
                labels=val_df['label'],
                scores=val_df['prediction_prob'],
            )
            labels = test_df['label'].astype(int)
            scores = test_df['prediction_prob'].astype(float)
            rows.append(
                {
                    'model': model_name,
                    'eval_type': 'test',
                    'fold_index': int(fold_index),
                    'eval_regime': regime,
                    'selected_threshold': threshold,
                    'balanced_accuracy_fixed_0_5': balanced_accuracy_score(
                        labels,
                        (scores >= 0.5).astype(int),
                    ),
                    'balanced_accuracy_val_tuned': balanced_accuracy_score(
                        labels,
                        (scores >= threshold).astype(int),
                    ),
                    'n_samples': len(test_df),
                }
            )

    threshold_df = pd.DataFrame(rows)
    average_rows = []
    for fold_index, group_df in threshold_df.groupby('fold_index'):
        average_rows.append(
            {
                'model': model_name,
                'eval_type': 'test',
                'fold_index': int(fold_index),
                'eval_regime': 'average',
                'selected_threshold': float(group_df['selected_threshold'].mean()),
                'balanced_accuracy_fixed_0_5': float(
                    group_df['balanced_accuracy_fixed_0_5'].mean()
                ),
                'balanced_accuracy_val_tuned': float(
                    group_df['balanced_accuracy_val_tuned'].mean()
                ),
                'n_samples': int(group_df['n_samples'].sum()),
            }
        )
    return pd.concat([threshold_df, pd.DataFrame(average_rows)], ignore_index=True)


def aggregate_threshold_metrics(threshold_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, eval_type, regime), group_df in threshold_fold_metrics.groupby(
        ['model', 'eval_type', 'eval_regime']
    ):
        row = {
            'model': model,
            'eval_type': eval_type,
            'eval_regime': regime,
            'n_folds': group_df['fold_index'].nunique(),
        }
        for metric_name in [
            'selected_threshold',
            'balanced_accuracy_fixed_0_5',
            'balanced_accuracy_val_tuned',
        ]:
            values = group_df[metric_name]
            mean_value = float(values.mean())
            sem_value = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            row[f'{metric_name}_mean'] = mean_value
            row[f'{metric_name}_sem'] = sem_value
            if metric_name.startswith('balanced_accuracy'):
                row[f'{metric_name}_display'] = (
                    f'{100 * mean_value:.1f} +/- {100 * sem_value:.1f}'
                )
            else:
                row[f'{metric_name}_display'] = f'{mean_value:.3f} +/- {sem_value:.3f}'
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['eval_type', 'model', 'eval_regime'])


def load_official_text_roberta(path: Path) -> str | None:
    if not path.exists():
        return None
    baseline_df = pd.read_csv(path)
    rows = baseline_df[baseline_df['Model'] == 'Text-Only Roberta']
    if rows.empty:
        return None
    return str(rows.iloc[0]['Average_AUROC'])


def write_report(
    output_root: Path,
    model_name: str,
    alpha_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    official_text_roberta: str | None,
) -> None:
    test_average = summary_df[
        (summary_df['eval_type'] == 'test')
        & (summary_df['eval_regime'] == 'average')
    ].iloc[0]
    val_average = summary_df[
        (summary_df['eval_type'] == 'val')
        & (summary_df['eval_regime'] == 'average')
    ].iloc[0]
    threshold_average = threshold_summary_df[
        (threshold_summary_df['eval_type'] == 'test')
        & (threshold_summary_df['eval_regime'] == 'average')
    ].iloc[0]

    lines = [
        f'# {model_name}',
        '',
        'Fold-local alpha is selected on validation AUROC, then applied unchanged to test predictions.',
        '',
        f"- Validation average AUROC: {val_average['auroc_display']}",
        f"- Test average AUROC: {test_average['auroc_display']}",
        '- Test average balanced accuracy with validation-tuned thresholds: '
        f"{threshold_average['balanced_accuracy_val_tuned_display']}",
    ]
    if official_text_roberta is not None:
        lines.append(f'- Official Text-only RoBERTa test average AUROC: {official_text_roberta}')
    lines.extend(
        [
            '',
            '## Selected Alphas',
            '',
            safe_to_markdown(alpha_df, index=False),
            '',
            '## Summary',
            '',
            safe_to_markdown(summary_df, index=False),
            '',
            '## Threshold Summary',
            '',
            safe_to_markdown(threshold_summary_df, index=False),
            '',
        ]
    )
    (output_root / 'summary' / 'late_fusion_report.md').write_text('\n'.join(lines))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_root = args.output_root / 'summary'
    summary_root.mkdir(parents=True, exist_ok=True)

    blended_frames = []
    alpha_rows = []
    for fold_index in args.folds:
        roberta_df = read_predictions(
            root=args.roberta_root,
            fold_index=fold_index,
            filename='trial_level_test_results.csv',
        )
        cec_df = read_predictions(
            root=args.cec_root,
            fold_index=fold_index,
            filename=args.cec_result_filename,
        )
        blended_df, alpha, val_auroc = blend_fold(
            roberta_df=roberta_df,
            cec_df=cec_df,
            alpha_step=args.alpha_grid_step,
        )
        blended_frames.append(blended_df)
        alpha_rows.append(
            {
                'fold_index': fold_index,
                'cec_alpha': alpha,
                'validation_average_auroc': val_auroc,
            }
        )

        fold_output_dir = args.output_root / f'fold_index={fold_index}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        blended_df.to_csv(fold_output_dir / 'trial_level_test_results.csv', index=False)

    all_predictions = pd.concat(blended_frames, ignore_index=True)
    alpha_df = pd.DataFrame(alpha_rows)
    fold_metrics = compute_fold_metrics(
        model_name=args.model_name,
        predictions=all_predictions,
    )
    summary_df = aggregate_metrics(fold_metrics)
    threshold_fold_metrics = compute_threshold_fold_metrics(
        model_name=args.model_name,
        predictions=all_predictions,
    )
    threshold_summary_df = aggregate_threshold_metrics(threshold_fold_metrics)

    all_predictions.to_csv(args.output_root / 'trial_level_test_results_all.csv', index=False)
    alpha_df.to_csv(summary_root / 'selected_alphas.csv', index=False)
    fold_metrics.to_csv(summary_root / 'fold_metrics.csv', index=False)
    summary_df.to_csv(summary_root / 'summary_metrics.csv', index=False)
    threshold_fold_metrics.to_csv(
        summary_root / 'threshold_fold_metrics.csv',
        index=False,
    )
    threshold_summary_df.to_csv(
        summary_root / 'threshold_summary_metrics.csv',
        index=False,
    )
    write_report(
        output_root=args.output_root,
        model_name=args.model_name,
        alpha_df=alpha_df,
        summary_df=summary_df,
        threshold_summary_df=threshold_summary_df,
        official_text_roberta=load_official_text_roberta(args.official_baseline_csv),
    )

    test_average = summary_df[
        (summary_df['eval_type'] == 'test')
        & (summary_df['eval_regime'] == 'average')
    ].iloc[0]
    print(
        f"{args.model_name} test average AUROC: "
        f"{test_average['auroc_mean']:.4f} +/- {test_average['auroc_sem']:.4f}"
    )
    print(f'Wrote blended outputs to {args.output_root}')


if __name__ == '__main__':
    main()
