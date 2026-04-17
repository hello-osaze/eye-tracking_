from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-eyebench')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
from loguru import logger

from src.run.multi_run.raw_to_processed_results import (
    DiscriSupportedMetrics,
    get_scores,
)
from src.run.single_run.report_tables import safe_to_markdown


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBERTA_ROOT = REPO_ROOT / 'results' / 'reference_raw_iitbhgc' / (
    '+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
    'trainer.wandb_job_type=Roberta_IITBHGC_CV'
)
DEFAULT_OFFICIAL_TEMPLATE = (
    REPO_ROOT / 'results' / 'formatted_eyebench_benchmark_results' / 'IITBHGC_CV_{eval_type}.csv'
)
DISPLAY_REGIME_ORDER = [
    'unseen_subject_seen_item',
    'seen_subject_unseen_item',
    'unseen_subject_unseen_item',
    'all',
]
DISPLAY_REGIME_LABELS = {
    'unseen_subject_seen_item': 'Unseen Reader',
    'seen_subject_unseen_item': 'Unseen Text',
    'unseen_subject_unseen_item': 'Unseen Text & Reader',
    'all': 'Average',
}
OFFICIAL_REGIME_LABELS = {
    'unseen_subject_seen_item': 'Unseen Reader',
    'seen_subject_unseen_item': 'Unseen Text',
    'unseen_subject_unseen_item': 'Unseen Text \\& Reader',
    'all': 'Average',
}
TEXT_ONLY_ROBERTA_MODEL_NAME = 'Text-Only Roberta'
BALANCED_ACCURACY_COLUMN_TEMPLATE = '{label}_\\makecell{{Balanced\\\\Accuracy}}'
AUROC_COLUMN_TEMPLATE = '{label}_AUROC'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify that the raw IITBHGC Text-only Roberta reference reproduces '
            'the official EyeBench formatted benchmark row.'
        )
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=DEFAULT_ROBERTA_ROOT,
        help='Root containing fold_index=*/trial_level_test_results.csv for the Text-only Roberta reference.',
    )
    parser.add_argument(
        '--official-template',
        type=Path,
        default=DEFAULT_OFFICIAL_TEMPLATE,
        help=(
            'Path template for the official formatted benchmark CSVs. Use '
            '`{eval_type}` in the filename.'
        ),
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('outputs/iitbhgc_roberta_reference_check'),
        help='Directory where the verification summary is written.',
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        help='Fold indices expected in the raw reference root.',
    )
    return parser.parse_args()


def resolve_template_path(template: Path, eval_type: str) -> Path:
    return Path(str(template).format(eval_type=eval_type))


def read_predictions(roberta_root: Path, folds: list[int]) -> pd.DataFrame:
    frames = []
    for fold_index in folds:
        path = roberta_root / f'fold_index={fold_index}' / 'trial_level_test_results.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    required_columns = {'label', 'prediction_prob', 'eval_type', 'eval_regime', 'fold_index'}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f'Raw reference is missing required columns: {missing_columns}')
    return df


def format_metric(values: list[float]) -> str:
    mean_value = 100.0 * float(np.mean(values))
    sem_value = 100.0 * float(np.std(values) / math.sqrt(len(values))) if values else 0.0
    return f'{mean_value:.1f} ± {sem_value:.1f}'


def summarize_eval_type(df: pd.DataFrame, eval_type: str) -> dict[str, str]:
    eval_df = df[df['eval_type'] == eval_type].copy()
    if eval_df.empty:
        raise ValueError(f'No rows found for eval_type={eval_type!r}')

    summary: dict[str, str] = {}
    for regime_name in DISPLAY_REGIME_ORDER:
        regime_scores_auroc: list[float] = []
        regime_scores_bacc: list[float] = []
        for fold_index in sorted(eval_df['fold_index'].unique()):
            if regime_name == 'all':
                fold_df = eval_df[eval_df['fold_index'] == fold_index]
            else:
                fold_df = eval_df[
                    (eval_df['fold_index'] == fold_index)
                    & (eval_df['eval_regime'] == regime_name)
                ]
            if fold_df.empty:
                raise ValueError(
                    f'No rows found for eval_type={eval_type!r}, fold={fold_index}, '
                    f'regime={regime_name!r}'
                )
            labels = fold_df['label']
            scores = fold_df['prediction_prob']
            regime_scores_auroc.append(
                get_scores(
                    y_true=labels,
                    prediction_prob=scores,
                    metric_name=DiscriSupportedMetrics.AUROC,
                )
            )
            regime_scores_bacc.append(
                get_scores(
                    y_true=labels,
                    prediction_prob=scores,
                    metric_name=DiscriSupportedMetrics.BALANCED_ACCURACY,
                )
            )

        label = DISPLAY_REGIME_LABELS[regime_name]
        summary[AUROC_COLUMN_TEMPLATE.format(label=label)] = format_metric(regime_scores_auroc)
        summary[BALANCED_ACCURACY_COLUMN_TEMPLATE.format(label=label)] = format_metric(
            regime_scores_bacc
        )
    return summary


def load_official_row(official_template: Path, eval_type: str) -> pd.Series:
    csv_path = resolve_template_path(official_template, eval_type=eval_type)
    official_df = pd.read_csv(csv_path)
    rows = official_df[official_df['Model'] == TEXT_ONLY_ROBERTA_MODEL_NAME]
    if rows.empty:
        raise ValueError(
            f'Could not find {TEXT_ONLY_ROBERTA_MODEL_NAME!r} in {csv_path}'
        )
    return rows.iloc[0]


def compare_to_official(
    eval_type: str,
    computed_summary: dict[str, str],
    official_row: pd.Series,
) -> list[tuple[str, str, str, str]]:
    mismatches = []
    for regime_name in DISPLAY_REGIME_ORDER:
        display_label = DISPLAY_REGIME_LABELS[regime_name]
        official_label = OFFICIAL_REGIME_LABELS[regime_name]
        columns = [
            (
                AUROC_COLUMN_TEMPLATE.format(label=display_label),
                AUROC_COLUMN_TEMPLATE.format(label=official_label),
            ),
            (
                BALANCED_ACCURACY_COLUMN_TEMPLATE.format(label=display_label),
                BALANCED_ACCURACY_COLUMN_TEMPLATE.format(label=official_label),
            ),
        ]
        for computed_column_name, official_column_name in columns:
            computed_value = computed_summary[computed_column_name]
            official_value = str(official_row[official_column_name])
            if computed_value != official_value:
                mismatches.append(
                    (eval_type, official_column_name, computed_value, official_value)
                )
    return mismatches


def build_report(
    output_root: Path,
    roberta_root: Path,
    val_summary: dict[str, str],
    test_summary: dict[str, str],
    val_row: pd.Series,
    test_row: pd.Series,
    mismatches: list[tuple[str, str, str, str]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    def as_table(summary: dict[str, str], official_row: pd.Series) -> pd.DataFrame:
        rows = []
        for regime_name in DISPLAY_REGIME_ORDER:
            label = DISPLAY_REGIME_LABELS[regime_name]
            official_label = OFFICIAL_REGIME_LABELS[regime_name]
            rows.append(
                {
                    'regime': label,
                    'computed_auroc': summary[AUROC_COLUMN_TEMPLATE.format(label=label)],
                    'official_auroc': str(
                        official_row[AUROC_COLUMN_TEMPLATE.format(label=official_label)]
                    ),
                    'computed_balanced_accuracy': summary[
                        BALANCED_ACCURACY_COLUMN_TEMPLATE.format(label=label)
                    ],
                    'official_balanced_accuracy': str(
                        official_row[
                            BALANCED_ACCURACY_COLUMN_TEMPLATE.format(label=official_label)
                        ]
                    ),
                }
            )
        return pd.DataFrame(rows)

    val_table = as_table(val_summary, val_row)
    test_table = as_table(test_summary, test_row)
    mismatch_df = pd.DataFrame(
        mismatches,
        columns=['eval_type', 'column', 'computed', 'official'],
    )
    val_table.to_csv(output_root / 'val_comparison.csv', index=False)
    test_table.to_csv(output_root / 'test_comparison.csv', index=False)
    mismatch_df.to_csv(output_root / 'mismatches.csv', index=False)

    lines = [
        '# IITBHGC Text-Only RoBERTa Reference Check',
        '',
        f"- result: `{'PASS' if not mismatches else 'FAIL'}`",
        f"- raw reference root: `{roberta_root}`",
        '',
        '## VAL',
        '',
        safe_to_markdown(val_table, index=False, floatfmt='.4f'),
        '',
        '## TEST',
        '',
        safe_to_markdown(test_table, index=False, floatfmt='.4f'),
        '',
    ]
    if mismatches:
        lines.extend(
            [
                '## Mismatches',
                '',
                safe_to_markdown(mismatch_df, index=False, floatfmt='.4f'),
                '',
            ]
        )
    (output_root / 'roberta_reference_check.md').write_text('\n'.join(lines) + '\n')


def main() -> int:
    args = parse_args()
    roberta_root = args.roberta_root.resolve()
    output_root = args.output_root.resolve()
    raw_df = read_predictions(roberta_root=roberta_root, folds=args.folds)

    val_summary = summarize_eval_type(raw_df, eval_type='val')
    test_summary = summarize_eval_type(raw_df, eval_type='test')
    val_row = load_official_row(args.official_template, eval_type='val')
    test_row = load_official_row(args.official_template, eval_type='test')

    mismatches = compare_to_official('val', val_summary, val_row)
    mismatches.extend(compare_to_official('test', test_summary, test_row))

    build_report(
        output_root=output_root,
        roberta_root=roberta_root,
        val_summary=val_summary,
        test_summary=test_summary,
        val_row=val_row,
        test_row=test_row,
        mismatches=mismatches,
    )

    if mismatches:
        logger.error(
            'Text-only Roberta reference check failed with {} mismatched fields.',
            len(mismatches),
        )
        for eval_type, column_name, computed_value, official_value in mismatches:
            logger.error(
                'Mismatch in {} [{}]: computed={} official={}',
                eval_type,
                column_name,
                computed_value,
                official_value,
            )
        return 1

    logger.info('Text-only Roberta reference check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
