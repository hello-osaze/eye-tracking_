from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-eyebench')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve  # noqa: E402

from src.run.single_run.report_tables import safe_to_markdown


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT.parents[1]
ASSET_ROOT = PROJECT_ROOT / 'submission_assets'
FIGURE_ROOT = ASSET_ROOT / 'figures'
TABLE_ROOT = ASSET_ROOT / 'tables'

REGIMES = [
    'seen_subject_unseen_item',
    'unseen_subject_seen_item',
    'unseen_subject_unseen_item',
]

RAW_IITBHGC_BASELINES = {
    'Text-Only Roberta (raw)': '+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_IITBHGC_CV',
    'RoBERTEye-W (raw)': '+data=IITBHGC_CV,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_IITBHGC_CV',
    'RoBERTEye-F (raw)': '+data=IITBHGC_CV,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_IITBHGC_CV',
    'MAG-Eye (raw)': '+data=IITBHGC_CV,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_IITBHGC_CV',
    'PostFusion-Eye (raw)': '+data=IITBHGC_CV,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_IITBHGC_CV',
}


@dataclass
class ModelSummary:
    name: str
    auroc_mean: float
    auroc_sem: float
    bacc_mean: float
    bacc_sem: float


def ensure_dirs() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build submission figures, tables, and significance summaries for the CEC study.'
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=Path(
            'results/raw/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
            'trainer.wandb_job_type=Roberta_IITBHGC_CV'
        ),
        help='Root containing raw fold-wise RoBERTa prediction CSVs.',
    )
    parser.add_argument(
        '--direct-root-main',
        type=Path,
        default=Path('outputs/cec_gaze_claim_context_mlp_mps_large_true_e10'),
        help='Primary root containing the trusted learned/no-scorer CEC study outputs.',
    )
    parser.add_argument(
        '--direct-root-ablation',
        type=Path,
        default=Path('outputs/cec_gaze_ablation_completion_mps_large_true_e10'),
        help='Optional secondary root used for missing direct ablations such as no-coverage and text-only.',
    )
    parser.add_argument(
        '--fusion-root',
        type=Path,
        default=Path('outputs/cec_roberta_late_fusion_mps_large_true_e10'),
        help='Root containing late-fusion prediction CSVs.',
    )
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray | list[float]) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def display(mean: float, error: float) -> str:
    return f'{100 * mean:.1f} +/- {100 * error:.1f}'


def read_fold_predictions(root: Path, filename: str = 'trial_level_test_results.csv') -> pd.DataFrame:
    frames = []
    for fold_index in [0, 1, 2, 3]:
        path = root / f'fold_index={fold_index}' / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    return df.loc[:, ~df.columns.duplicated()].copy()


def select_threshold(labels: pd.Series, scores: pd.Series) -> float:
    if labels.nunique() < 2 or scores.nunique() < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels.astype(int), scores.astype(float))
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return 0.5
    j_scores = tpr[finite_mask] - fpr[finite_mask]
    return float(thresholds[finite_mask][int(np.argmax(j_scores))])


def summarize_trial_predictions(model_name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    threshold_rows = []
    for (eval_type, fold_index, regime), group_df in df.groupby(
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
                'balanced_accuracy_fixed_0_5': balanced_accuracy_score(
                    labels,
                    (scores >= 0.5).astype(int),
                ),
                'n_samples': len(group_df),
            }
        )

    per_regime_df = pd.DataFrame(rows)
    for fold_index in sorted(df['fold_index'].unique()):
        for regime in REGIMES:
            val_df = df[
                (df['eval_type'] == 'val')
                & (df['fold_index'] == fold_index)
                & (df['eval_regime'] == regime)
            ]
            test_df = df[
                (df['eval_type'] == 'test')
                & (df['fold_index'] == fold_index)
                & (df['eval_regime'] == regime)
            ]
            threshold = select_threshold(
                labels=val_df['label'],
                scores=val_df['prediction_prob'],
            )
            threshold_rows.append(
                {
                    'model': model_name,
                    'eval_type': 'test',
                    'fold_index': int(fold_index),
                    'eval_regime': regime,
                    'selected_threshold': threshold,
                    'balanced_accuracy_val_tuned': balanced_accuracy_score(
                        test_df['label'].astype(int),
                        (test_df['prediction_prob'].astype(float) >= threshold).astype(int),
                    ),
                }
            )

    threshold_df = pd.DataFrame(threshold_rows)
    return per_regime_df, threshold_df


def aggregate_summary(per_regime_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    average_rows = []
    for (model, eval_type, fold_index), group_df in per_regime_df.groupby(
        ['model', 'eval_type', 'fold_index']
    ):
        average_rows.append(
            {
                'model': model,
                'eval_type': eval_type,
                'fold_index': fold_index,
                'eval_regime': 'average',
                'auroc': float(group_df['auroc'].mean()),
                'balanced_accuracy_fixed_0_5': float(
                    group_df['balanced_accuracy_fixed_0_5'].mean()
                ),
            }
        )
    fold_df = pd.concat([per_regime_df, pd.DataFrame(average_rows)], ignore_index=True)

    threshold_average_rows = []
    for (model, fold_index), group_df in threshold_df.groupby(['model', 'fold_index']):
        threshold_average_rows.append(
            {
                'model': model,
                'eval_type': 'test',
                'fold_index': fold_index,
                'eval_regime': 'average',
                'selected_threshold': float(group_df['selected_threshold'].mean()),
                'balanced_accuracy_val_tuned': float(
                    group_df['balanced_accuracy_val_tuned'].mean()
                ),
            }
        )
    threshold_fold_df = pd.concat(
        [threshold_df, pd.DataFrame(threshold_average_rows)],
        ignore_index=True,
    )

    rows = []
    for (model, eval_type, regime), group_df in fold_df.groupby(
        ['model', 'eval_type', 'eval_regime']
    ):
        row = {
            'model': model,
            'eval_type': eval_type,
            'eval_regime': regime,
            'n_folds': int(group_df['fold_index'].nunique()),
            'auroc_mean': float(group_df['auroc'].mean()),
            'auroc_sem': sem(group_df['auroc']),
            'balanced_accuracy_fixed_0_5_mean': float(
                group_df['balanced_accuracy_fixed_0_5'].mean()
            ),
            'balanced_accuracy_fixed_0_5_sem': sem(
                group_df['balanced_accuracy_fixed_0_5']
            ),
        }
        tuned = threshold_fold_df[
            (threshold_fold_df['model'] == model)
            & (threshold_fold_df['eval_type'] == eval_type)
            & (threshold_fold_df['eval_regime'] == regime)
        ]
        row['balanced_accuracy_val_tuned_mean'] = float(
            tuned['balanced_accuracy_val_tuned'].mean()
        )
        row['balanced_accuracy_val_tuned_sem'] = sem(
            tuned['balanced_accuracy_val_tuned']
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['model', 'eval_type', 'eval_regime'])


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    merge_keys = ['fold_index', 'eval_type', 'eval_regime', 'unique_trial_id']
    merged = df_a[merge_keys + ['label', 'prediction_prob']].merge(
        df_b[merge_keys + ['label', 'prediction_prob']],
        on=merge_keys,
        suffixes=('_a', '_b'),
        how='inner',
    )
    merged = merged[merged['eval_type'] == 'test'].copy()
    if not (merged['label_a'].astype(int) == merged['label_b'].astype(int)).all():
        raise ValueError('Labels disagree after merge.')

    grouped = []
    for (_, fold_index, regime), group_df in merged.groupby(
        ['eval_type', 'fold_index', 'eval_regime']
    ):
        labels = group_df['label_a'].astype(int).to_numpy()
        scores_a = group_df['prediction_prob_a'].astype(float).to_numpy()
        scores_b = group_df['prediction_prob_b'].astype(float).to_numpy()
        grouped.append((int(fold_index), regime, labels, scores_a, scores_b))

    def mean_delta_from_indices(indices_per_group: list[np.ndarray] | None) -> float:
        deltas = []
        for group_index, (_, _, labels, scores_a, scores_b) in enumerate(grouped):
            if indices_per_group is None:
                idx = np.arange(len(labels))
            else:
                idx = indices_per_group[group_index]
            auc_a = roc_auc_score(labels[idx], scores_a[idx])
            auc_b = roc_auc_score(labels[idx], scores_b[idx])
            deltas.append(auc_a - auc_b)
        return float(np.mean(deltas))

    observed = mean_delta_from_indices(indices_per_group=None)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        indices_per_group = [
            rng.integers(0, len(labels), len(labels))
            for _, _, labels, _, _ in grouped
        ]
        samples.append(mean_delta_from_indices(indices_per_group))
    sample_series = pd.Series(samples, dtype=float)
    p_one_sided = (
        float((sample_series <= 0).mean())
        if observed >= 0
        else float((sample_series >= 0).mean())
    )
    return {
        'delta_mean': observed,
        'delta_sem_boot': sem(sample_series),
        'delta_ci_low': float(sample_series.quantile(0.025)),
        'delta_ci_high': float(sample_series.quantile(0.975)),
        'p_one_sided': p_one_sided,
        'p_two_sided': min(1.0, 2.0 * p_one_sided),
    }


def paired_bootstrap_mean_difference(
    values_a: pd.Series | np.ndarray,
    values_b: pd.Series | np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    a = pd.Series(values_a, dtype=float).to_numpy()
    b = pd.Series(values_b, dtype=float).to_numpy()
    if len(a) != len(b):
        raise ValueError('Paired arrays must have the same length.')
    diffs = a - b
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diffs), len(diffs))
        samples.append(float(diffs[idx].mean()))
    sample_series = pd.Series(samples, dtype=float)
    p_one_sided = (
        float((sample_series <= 0).mean())
        if observed >= 0
        else float((sample_series >= 0).mean())
    )
    return {
        'delta_mean': observed,
        'delta_sem_boot': sem(sample_series),
        'delta_ci_low': float(sample_series.quantile(0.025)),
        'delta_ci_high': float(sample_series.quantile(0.975)),
        'p_one_sided': p_one_sided,
        'p_two_sided': min(1.0, 2.0 * p_one_sided),
    }


def load_reference_baselines() -> pd.DataFrame:
    path = REPO_ROOT / 'results' / 'formatted_eyebench_benchmark_results' / 'IITBHGC_CV_test.csv'
    df = pd.read_csv(path)
    keep = [
        'Text-Only Roberta',
        'RoBERTEye-W~\\citep{Shubi2024Finegrained}',
        'RoBERTEye-F~\\citep{Shubi2024Finegrained}',
        'MAG-Eye~\\citep{Shubi2024Finegrained}',
        'PostFusion-Eye~\\citep{Shubi2024Finegrained}',
    ]
    return df[df['Model'].isin(keep)].copy()


def parse_pm(value: str) -> tuple[float, float]:
    mean_text, sem_text = value.replace('±', '+/-').split('+/-')
    return float(mean_text.strip()) / 100.0, float(sem_text.strip()) / 100.0


def plot_benchmark_auroc(benchmark_df: pd.DataFrame, ours_df: pd.DataFrame) -> None:
    rows = []
    for _, row in benchmark_df.iterrows():
        mean_value, sem_value = parse_pm(row['Average_AUROC'])
        rows.append(
            {
                'label': row['Model'].replace('~\\citep{Shubi2024Finegrained}', '').replace('\\cite{meziere2023using}', ''),
                'auroc_mean': mean_value,
                'auroc_sem': sem_value,
                'group': 'official',
            }
        )
    rows.extend(ours_df.to_dict(orient='records'))
    plot_df = pd.DataFrame(rows)

    colors = [
        '#6c7a89' if row['group'] == 'official' else '#0f766e'
        for _, row in plot_df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(plot_df))
    ax.bar(
        x,
        100 * plot_df['auroc_mean'],
        yerr=100 * plot_df['auroc_sem'],
        color=colors,
        alpha=0.9,
        capsize=4,
    )
    ax.set_ylabel('Test AUROC')
    ax.set_ylim(52, 61)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['label'], rotation=25, ha='right')
    ax.set_title('Benchmark Comparison on IITBHGC')
    ax.axhline(58.8, color='#9ca3af', linestyle='--', linewidth=1)
    ax.text(len(plot_df) - 1.1, 58.95, 'Text-only RoBERTa benchmark', color='#6b7280')
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / 'benchmark_auroc.png', dpi=220)
    plt.close(fig)


def plot_ablations(direct_df: pd.DataFrame, fusion_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    for ax, plot_df, title in [
        (axes[0], direct_df, 'Direct CEC Variants'),
        (axes[1], fusion_df, 'Late-Fusion Variants'),
    ]:
        x = np.arange(len(plot_df))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_df)))
        ax.bar(
            x,
            100 * plot_df['auroc_mean'],
            yerr=100 * plot_df['auroc_sem'],
            color=colors,
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['label'], rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylabel('Test AUROC')
        ax.set_ylim(55, 60)

    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / 'ablation_auroc.png', dpi=220)
    plt.close(fig)


def plot_regime_gains(roberta_summary: pd.DataFrame, fusion_summary: pd.DataFrame) -> None:
    regimes = REGIMES + ['average']
    rows = []
    for regime in regimes:
        base = roberta_summary[
            (roberta_summary['eval_type'] == 'test')
            & (roberta_summary['eval_regime'] == regime)
        ].iloc[0]
        fused = fusion_summary[
            (fusion_summary['eval_type'] == 'test')
            & (fusion_summary['eval_regime'] == regime)
        ].iloc[0]
        rows.append(
            {
                'regime': regime.replace('_', '\n'),
                'delta': 100 * (fused['auroc_mean'] - base['auroc_mean']),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        plot_df['regime'],
        plot_df['delta'],
        color=['#0f766e' if value >= 0 else '#dc2626' for value in plot_df['delta']],
    )
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_ylabel('AUROC gain over raw RoBERTa (points)')
    ax.set_title('Where Late Fusion Gains Appear')
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / 'regime_gains.png', dpi=220)
    plt.close(fig)


def plot_score_drop(score_drop_path: Path) -> None:
    df = pd.concat(
        [pd.read_csv(path) for path in sorted(score_drop_path.glob('fold_index=*/score_drop_summary.csv'))],
        ignore_index=True,
    )
    fold_df = (
        df.groupby('fold_index', as_index=False)[
            ['mean_abs_delta_top_drop', 'mean_abs_delta_random_drop']
        ]
        .mean()
    )
    means = [
        float(fold_df['mean_abs_delta_top_drop'].mean()),
        float(fold_df['mean_abs_delta_random_drop'].mean()),
    ]
    errors = [
        sem(fold_df['mean_abs_delta_top_drop']),
        sem(fold_df['mean_abs_delta_random_drop']),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.bar(
        ['Top-score drop', 'Random drop'],
        means,
        yerr=errors,
        color=['#0f766e', '#9ca3af'],
        capsize=4,
    )
    ax.set_ylabel('Mean abs probability change')
    ax.set_title('Score-Drop Sensitivity')
    ratio = means[0] / means[1]
    ax.text(0.45, max(means) * 0.92, f'~{ratio:.1f}x larger', ha='center')
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / 'score_drop.png', dpi=220)
    plt.close(fig)


def summarize_score_drop_trials(score_drop_path: Path) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(score_drop_path.glob('fold_index=*/score_drop_test_results.csv'))
        ],
        ignore_index=True,
    )
    df['abs_delta_top_drop'] = df['prediction_prob_delta_top_drop'].abs()
    df['abs_delta_random_drop'] = df['prediction_prob_delta_random_drop_mean'].abs()

    rows = []
    for regime_name, group_df in [('average', df), *list(df.groupby('eval_regime'))]:
        stats = paired_bootstrap_mean_difference(
            values_a=group_df['abs_delta_top_drop'],
            values_b=group_df['abs_delta_random_drop'],
        )
        mean_top = float(group_df['abs_delta_top_drop'].mean())
        mean_random = float(group_df['abs_delta_random_drop'].mean())
        rows.append(
            {
                'eval_regime': regime_name,
                'n_trials': int(len(group_df)),
                'mean_abs_delta_top_drop': mean_top,
                'mean_abs_delta_random_drop': mean_random,
                'mean_abs_delta_difference': stats['delta_mean'],
                'difference_ci_low': stats['delta_ci_low'],
                'difference_ci_high': stats['delta_ci_high'],
                'p_one_sided': stats['p_one_sided'],
                'p_two_sided': stats['p_two_sided'],
                'top_over_random_ratio': mean_top / mean_random,
                'fraction_top_gt_random': float(
                    (
                        group_df['abs_delta_top_drop']
                        > group_df['abs_delta_random_drop']
                    ).mean()
                ),
                'mean_removed_top_evidence_mass': float(
                    group_df['removed_top_evidence_mass'].mean()
                ),
                'mean_n_dropped_tokens': float(group_df['n_dropped_tokens'].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    roberta_root = REPO_ROOT / args.roberta_root
    direct_root_main = REPO_ROOT / args.direct_root_main
    direct_root_ablation = REPO_ROOT / args.direct_root_ablation
    fusion_root = REPO_ROOT / args.fusion_root

    results_raw_root = roberta_root.parent
    raw_baseline_frames = {}
    raw_baseline_summary_frames = []
    for model_name, subdir in RAW_IITBHGC_BASELINES.items():
        root = results_raw_root / subdir
        if not (root / 'fold_index=0' / 'trial_level_test_results.csv').exists():
            continue
        df = read_fold_predictions(root)
        raw_baseline_frames[model_name] = df
        per_regime_df, threshold_df = summarize_trial_predictions(model_name=model_name, df=df)
        raw_baseline_summary_frames.append(
            aggregate_summary(per_regime_df=per_regime_df, threshold_df=threshold_df)
        )
    raw_baseline_summary = pd.concat(raw_baseline_summary_frames, ignore_index=True)
    raw_roberta_df = raw_baseline_frames['Text-Only Roberta (raw)']
    raw_roberta_summary = raw_baseline_summary[
        raw_baseline_summary['model'] == 'Text-Only Roberta (raw)'
    ].copy()

    direct_specs = {
        'CECGaze': (direct_root_main / 'CECGaze', 'trial_level_test_results.csv'),
    }
    optional_direct_specs = {
        'CECGazeNoScorer': [
            (direct_root_main / 'CECGazeNoScorer', 'trial_level_test_results.csv'),
        ],
        'CECGazeUniformEval': [
            (direct_root_main / 'CECGaze', 'trial_level_test_results_uniform.csv'),
        ],
        'CECGazeShuffleEval': [
            (direct_root_main / 'CECGaze', 'trial_level_test_results_shuffle.csv'),
        ],
        'CECGazeNoCoverage': [
            (direct_root_ablation / 'CECGazeNoCoverage', 'trial_level_test_results.csv'),
            (direct_root_main / 'CECGazeNoCoverage', 'trial_level_test_results.csv'),
        ],
        'CECGazeTextOnly': [
            (direct_root_ablation / 'CECGazeTextOnly', 'trial_level_test_results.csv'),
            (direct_root_main / 'CECGazeTextOnly', 'trial_level_test_results.csv'),
        ],
        'CECGazeZeroCoverageEval': [
            (direct_root_main / 'CECGaze', 'trial_level_test_results_zero_coverage.csv'),
        ],
        'CECGazeZeroGazeEval': [
            (direct_root_main / 'CECGaze', 'trial_level_test_results_zero_gaze.csv'),
        ],
    }
    for name, candidate_specs in optional_direct_specs.items():
        for root, filename in candidate_specs:
            if (root / 'fold_index=0' / filename).exists():
                direct_specs[name] = (root, filename)
                break
    direct_frames = {}
    direct_summary_frames = []
    for name, (root, filename) in direct_specs.items():
        df = read_fold_predictions(root, filename=filename)
        direct_frames[name] = df
        per_regime_df, threshold_df = summarize_trial_predictions(name, df)
        direct_summary_frames.append(
            aggregate_summary(per_regime_df=per_regime_df, threshold_df=threshold_df)
        )
    direct_summary = pd.concat(direct_summary_frames, ignore_index=True)

    fusion_specs = {
        'CECGaze+RoBERTa': fusion_root / 'CECGazeRobertaValBlendFine' / 'trial_level_test_results_all.csv',
    }
    optional_fusion_specs = {
        'NoScorer+RoBERTa': fusion_root / 'CECGazeNoScorerRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'Uniform+RoBERTa': fusion_root / 'CECGazeUniformRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'Shuffle+RoBERTa': fusion_root / 'CECGazeShuffleRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'NoCoverage+RoBERTa': fusion_root / 'CECGazeNoCoverageRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'TextOnly+RoBERTa': fusion_root / 'CECGazeTextOnlyRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'ZeroCoverageEval+RoBERTa': fusion_root / 'CECGazeZeroCoverageRobertaValBlendFine' / 'trial_level_test_results_all.csv',
        'ZeroGazeEval+RoBERTa': fusion_root / 'CECGazeZeroGazeRobertaValBlendFine' / 'trial_level_test_results_all.csv',
    }
    for name, path in optional_fusion_specs.items():
        if path.exists():
            fusion_specs[name] = path
    fusion_frames = {}
    fusion_summary_frames = []
    for name, path in fusion_specs.items():
        df = pd.read_csv(path)
        fusion_frames[name] = df
        per_regime_df, threshold_df = summarize_trial_predictions(name, df)
        fusion_summary_frames.append(
            aggregate_summary(per_regime_df=per_regime_df, threshold_df=threshold_df)
        )
    fusion_summary = pd.concat(fusion_summary_frames, ignore_index=True)

    comparisons = [
        ('CECGaze+RoBERTa', 'Text-Only Roberta (raw)', fusion_frames['CECGaze+RoBERTa'], raw_roberta_df),
    ]
    for model_name in ['NoScorer+RoBERTa', 'Uniform+RoBERTa', 'Shuffle+RoBERTa']:
        if model_name in fusion_frames:
            comparisons.append(
                (
                    'CECGaze+RoBERTa',
                    model_name,
                    fusion_frames['CECGaze+RoBERTa'],
                    fusion_frames[model_name],
                )
            )
    for model_name in ['CECGazeNoScorer', 'CECGazeUniformEval', 'CECGazeShuffleEval']:
        if model_name in direct_frames:
            comparisons.append(
                (
                    'CECGaze',
                    model_name,
                    direct_frames['CECGaze'],
                    direct_frames[model_name],
                )
            )
    if 'NoCoverage+RoBERTa' in fusion_frames:
        comparisons.append(
            ('CECGaze+RoBERTa', 'NoCoverage+RoBERTa', fusion_frames['CECGaze+RoBERTa'], fusion_frames['NoCoverage+RoBERTa'])
        )
    if 'TextOnly+RoBERTa' in fusion_frames:
        comparisons.append(
            ('CECGaze+RoBERTa', 'TextOnly+RoBERTa', fusion_frames['CECGaze+RoBERTa'], fusion_frames['TextOnly+RoBERTa'])
        )
    if 'ZeroCoverageEval+RoBERTa' in fusion_frames:
        comparisons.append(
            (
                'CECGaze+RoBERTa',
                'ZeroCoverageEval+RoBERTa',
                fusion_frames['CECGaze+RoBERTa'],
                fusion_frames['ZeroCoverageEval+RoBERTa'],
            )
        )
    if 'ZeroGazeEval+RoBERTa' in fusion_frames:
        comparisons.append(
            (
                'CECGaze+RoBERTa',
                'ZeroGazeEval+RoBERTa',
                fusion_frames['CECGaze+RoBERTa'],
                fusion_frames['ZeroGazeEval+RoBERTa'],
            )
        )
    if 'CECGazeNoCoverage' in direct_frames:
        comparisons.append(
            ('CECGaze', 'CECGazeNoCoverage', direct_frames['CECGaze'], direct_frames['CECGazeNoCoverage'])
        )
    if 'CECGazeTextOnly' in direct_frames:
        comparisons.append(
            ('CECGaze', 'CECGazeTextOnly', direct_frames['CECGaze'], direct_frames['CECGazeTextOnly'])
        )
    if 'CECGazeZeroCoverageEval' in direct_frames:
        comparisons.append(
            (
                'CECGaze',
                'CECGazeZeroCoverageEval',
                direct_frames['CECGaze'],
                direct_frames['CECGazeZeroCoverageEval'],
            )
        )
    if 'CECGazeZeroGazeEval' in direct_frames:
        comparisons.append(
            (
                'CECGaze',
                'CECGazeZeroGazeEval',
                direct_frames['CECGaze'],
                direct_frames['CECGazeZeroGazeEval'],
            )
        )
    for baseline_name in [
        'RoBERTEye-W (raw)',
        'RoBERTEye-F (raw)',
        'MAG-Eye (raw)',
        'PostFusion-Eye (raw)',
    ]:
        if baseline_name in raw_baseline_frames:
            comparisons.append(
                (
                    'CECGaze+RoBERTa',
                    baseline_name,
                    fusion_frames['CECGaze+RoBERTa'],
                    raw_baseline_frames[baseline_name],
                )
            )
    bootstrap_rows = []
    for model_a, model_b, df_a, df_b in comparisons:
        result = paired_bootstrap_delta(df_a=df_a, df_b=df_b)
        bootstrap_rows.append({'model_a': model_a, 'model_b': model_b, **result})
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    direct_avg = direct_summary[
        (direct_summary['eval_type'] == 'test') & (direct_summary['eval_regime'] == 'average')
    ].copy()
    fusion_avg = fusion_summary[
        (fusion_summary['eval_type'] == 'test') & (fusion_summary['eval_regime'] == 'average')
    ].copy()
    raw_avg = raw_roberta_summary[
        (raw_roberta_summary['eval_type'] == 'test') & (raw_roberta_summary['eval_regime'] == 'average')
    ].copy()

    benchmark_table = pd.concat([raw_avg, direct_avg, fusion_avg], ignore_index=True)
    benchmark_table.to_csv(TABLE_ROOT / 'benchmark_metrics.csv', index=False)
    bootstrap_df.to_csv(TABLE_ROOT / 'bootstrap_significance.csv', index=False)
    direct_summary.to_csv(TABLE_ROOT / 'direct_ablation_metrics.csv', index=False)
    fusion_summary.to_csv(TABLE_ROOT / 'fusion_ablation_metrics.csv', index=False)
    raw_baseline_summary.to_csv(TABLE_ROOT / 'official_raw_baseline_metrics.csv', index=False)
    raw_roberta_summary.to_csv(TABLE_ROOT / 'raw_roberta_metrics.csv', index=False)

    official_df = load_reference_baselines()
    ours_plot_rows = pd.DataFrame(
        [
            {
                'label': 'CECGaze direct',
                'auroc_mean': float(
                    direct_avg[direct_avg['model'] == 'CECGaze']['auroc_mean'].iloc[0]
                ),
                'auroc_sem': float(
                    direct_avg[direct_avg['model'] == 'CECGaze']['auroc_sem'].iloc[0]
                ),
                'group': 'ours',
            },
            {
                'label': 'CECGaze + RoBERTa',
                'auroc_mean': float(
                    fusion_avg[fusion_avg['model'] == 'CECGaze+RoBERTa']['auroc_mean'].iloc[0]
                ),
                'auroc_sem': float(
                    fusion_avg[fusion_avg['model'] == 'CECGaze+RoBERTa']['auroc_sem'].iloc[0]
                ),
                'group': 'ours',
            },
        ]
    )
    plot_benchmark_auroc(benchmark_df=official_df, ours_df=ours_plot_rows)

    direct_plot_order = [
        ('CECGaze', 'Learned'),
        ('CECGazeNoScorer', 'No scorer'),
        ('CECGazeNoCoverage', 'No coverage'),
        ('CECGazeTextOnly', 'Text only'),
        ('CECGazeZeroCoverageEval', 'Zero coverage (eval)'),
        ('CECGazeZeroGazeEval', 'Zero gaze (eval)'),
        ('CECGazeUniformEval', 'Uniform'),
        ('CECGazeShuffleEval', 'Shuffle'),
    ]
    direct_plot_rows = []
    for model_name, label in direct_plot_order:
        subset = direct_avg[direct_avg['model'] == model_name]
        if subset.empty:
            continue
        direct_plot_rows.append(
            {
                'label': label,
                'auroc_mean': float(subset['auroc_mean'].iloc[0]),
                'auroc_sem': float(subset['auroc_sem'].iloc[0]),
            }
        )
    direct_plot_df = pd.DataFrame(direct_plot_rows)

    fusion_plot_order = [
        ('CECGaze+RoBERTa', 'Learned'),
        ('NoScorer+RoBERTa', 'No scorer'),
        ('NoCoverage+RoBERTa', 'No coverage'),
        ('TextOnly+RoBERTa', 'Text only'),
        ('ZeroCoverageEval+RoBERTa', 'Zero coverage (eval)'),
        ('ZeroGazeEval+RoBERTa', 'Zero gaze (eval)'),
        ('Uniform+RoBERTa', 'Uniform'),
        ('Shuffle+RoBERTa', 'Shuffle'),
    ]
    fusion_plot_rows = []
    for model_name, label in fusion_plot_order:
        subset = fusion_avg[fusion_avg['model'] == model_name]
        if subset.empty:
            continue
        fusion_plot_rows.append(
            {
                'label': label,
                'auroc_mean': float(subset['auroc_mean'].iloc[0]),
                'auroc_sem': float(subset['auroc_sem'].iloc[0]),
            }
        )
    fusion_plot_df = pd.DataFrame(fusion_plot_rows)
    plot_ablations(direct_df=direct_plot_df, fusion_df=fusion_plot_df)
    plot_regime_gains(
        roberta_summary=raw_roberta_summary,
        fusion_summary=fusion_summary[fusion_summary['model'] == 'CECGaze+RoBERTa'],
    )
    plot_score_drop(score_drop_path=direct_root_main / 'CECGaze')
    score_drop_trial_df = summarize_score_drop_trials(score_drop_path=direct_root_main / 'CECGaze')
    score_drop_trial_df.to_csv(TABLE_ROOT / 'score_drop_trial_stats.csv', index=False)

    summary_lines = [
        '# Submission Asset Summary',
        '',
        '## Key Numbers',
        '',
    ]
    learned_fusion = fusion_avg[fusion_avg['model'] == 'CECGaze+RoBERTa'].iloc[0]
    raw_roberta = raw_avg.iloc[0]
    learned_direct = direct_avg[direct_avg['model'] == 'CECGaze'].iloc[0]
    summary_lines.extend(
        [
            f"- Raw text-only RoBERTa AUROC: {display(raw_roberta['auroc_mean'], raw_roberta['auroc_sem'])}",
            f"- Direct CECGaze AUROC: {display(learned_direct['auroc_mean'], learned_direct['auroc_sem'])}",
            f"- Late fusion AUROC: {display(learned_fusion['auroc_mean'], learned_fusion['auroc_sem'])}",
            '',
            '## Re-aggregated Raw Official Baselines',
            '',
            safe_to_markdown(
                raw_baseline_summary[
                    (raw_baseline_summary['eval_type'] == 'test')
                    & (raw_baseline_summary['eval_regime'] == 'average')
                ][
                    [
                        'model',
                        'auroc_mean',
                        'auroc_sem',
                        'balanced_accuracy_val_tuned_mean',
                        'balanced_accuracy_val_tuned_sem',
                    ]
                ],
                index=False,
                floatfmt='.4f',
            ),
            '',
            '## Bootstrap Comparisons',
            '',
            safe_to_markdown(bootstrap_df, index=False, floatfmt='.4f'),
            '',
            '## Score-Drop Trial Statistics',
            '',
            safe_to_markdown(score_drop_trial_df, index=False, floatfmt='.4f'),
            '',
            '## Figures',
            '',
            '- `figures/benchmark_auroc.png`',
            '- `figures/ablation_auroc.png`',
            '- `figures/regime_gains.png`',
            '- `figures/score_drop.png`',
            '',
        ]
    )
    (ASSET_ROOT / 'README.md').write_text('\n'.join(summary_lines) + '\n')


if __name__ == '__main__':
    main()
