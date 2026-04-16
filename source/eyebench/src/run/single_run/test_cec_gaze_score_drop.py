"""Evaluate whether CEC-Gaze predictions change when top-scored evidence tokens are removed."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import lightning_fabric as lf
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from src.configs.constants import REGIMES
from src.data.datamodules.base_datamodule import DataModuleFactory
from src.run.multi_run import supported_datamodules, supported_models  # noqa: F401
from src.run.single_run.cec_eval_utils import (
    choose_device,
    get_fold_paths,
    load_cec_gaze_model,
    move_batch_to_device,
)
from src.run.single_run.utils import (
    extract_trial_info,
)


@dataclass
class ScoreDropBatchResult:
    labels: torch.Tensor
    base_probs: torch.Tensor
    top_drop_probs: torch.Tensor
    random_drop_probs_mean: torch.Tensor
    random_drop_probs_std: torch.Tensor
    top_drop_delta: torch.Tensor
    random_drop_delta_mean: torch.Tensor
    removed_top_mass: torch.Tensor
    num_dropped_tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run CEC-Gaze score-drop controls on trained fold checkpoints.'
    )
    parser.add_argument(
        '--eval-path',
        required=True,
        type=Path,
        help='Path containing fold_index=* subdirectories with trained checkpoints.',
    )
    parser.add_argument(
        '--checkpoint-template',
        default='*lowest_loss_val_all*.ckpt',
        help='Checkpoint filename glob to select per fold.',
    )
    parser.add_argument(
        '--drop-fraction',
        default=0.2,
        type=float,
        help='Fraction of valid context tokens to remove per trial.',
    )
    parser.add_argument(
        '--random-repeats',
        default=10,
        type=int,
        help='Number of matched random-drop replicates per batch.',
    )
    parser.add_argument(
        '--batch-size',
        default=None,
        type=int,
        help='Optional evaluation batch size override for score-drop dataloaders.',
    )
    parser.add_argument(
        '--eval-types',
        nargs='+',
        default=['val', 'test'],
        choices=['val', 'test'],
        help='Which dataset splits to evaluate for score-drop.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        help='Device to use for evaluation.',
    )
    parser.add_argument(
        '--num-workers',
        default=None,
        type=int,
        help='Optional DataLoader worker override for evaluation.',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='Random seed for random-drop controls.',
    )
    parser.add_argument(
        '--output-name',
        default='score_drop_test_results.csv',
        help='Filename for per-trial score-drop outputs saved under each fold.',
    )
    parser.add_argument(
        '--summary-name',
        default='score_drop_summary.csv',
        help='Filename for grouped score-drop summary metrics saved under each fold.',
    )
    args = parser.parse_args()
    if not (0.0 < args.drop_fraction <= 1.0):
        raise ValueError(
            f'--drop-fraction must be in (0, 1], got {args.drop_fraction}.'
        )
    if args.random_repeats < 1:
        raise ValueError(
            f'--random-repeats must be >= 1, got {args.random_repeats}.'
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f'--batch-size must be >= 1, got {args.batch_size}.')
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError(f'--num-workers must be >= 0, got {args.num_workers}.')
    return args


def get_context_mask(batch_data, attention_mask: torch.Tensor) -> torch.Tensor:
    context_token_masks = getattr(batch_data, 'context_token_masks', None)
    if context_token_masks is None:
        context_token_masks = attention_mask.clone()
        context_token_masks[:, 0] = 0
    return context_token_masks.bool() & attention_mask.bool()


def get_positive_class_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(dim=1)
    return probs[:, 1] if probs.shape[1] == 2 else probs.max(dim=1).values


def get_class_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1 and labels.shape[-1] > 1:
        labels = labels.argmax(dim=-1)
    return labels.reshape(-1)


def choose_num_drop_tokens(valid_count: int, drop_fraction: float) -> int:
    if valid_count <= 0:
        return 0
    return max(1, min(valid_count, int(math.ceil(valid_count * drop_fraction))))


def build_top_drop_mask(
    evidence_weights: torch.Tensor,
    context_mask: torch.Tensor,
    drop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drop_mask = torch.zeros_like(context_mask)
    removed_mass = torch.zeros(
        evidence_weights.size(0),
        device=evidence_weights.device,
        dtype=evidence_weights.dtype,
    )
    num_dropped = torch.zeros(
        evidence_weights.size(0),
        device=evidence_weights.device,
        dtype=torch.long,
    )

    masked_weights = evidence_weights.masked_fill(~context_mask, float('-inf'))
    for row_idx in range(evidence_weights.size(0)):
        valid_indices = torch.nonzero(context_mask[row_idx], as_tuple=False).squeeze(-1)
        n_drop = choose_num_drop_tokens(
            valid_count=valid_indices.numel(),
            drop_fraction=drop_fraction,
        )
        if n_drop == 0:
            continue
        selected_indices = torch.topk(masked_weights[row_idx], k=n_drop).indices
        drop_mask[row_idx, selected_indices] = True
        removed_mass[row_idx] = evidence_weights[row_idx, selected_indices].sum()
        num_dropped[row_idx] = n_drop

    return drop_mask, removed_mass, num_dropped


def build_random_drop_mask(
    context_mask: torch.Tensor,
    num_dropped_tokens: torch.Tensor,
) -> torch.Tensor:
    drop_mask = torch.zeros_like(context_mask)
    for row_idx in range(context_mask.size(0)):
        valid_indices = torch.nonzero(context_mask[row_idx], as_tuple=False).squeeze(-1)
        n_drop = int(num_dropped_tokens[row_idx].item())
        if n_drop <= 0 or valid_indices.numel() == 0:
            continue
        n_drop = min(n_drop, valid_indices.numel())
        shuffled = valid_indices[
            torch.randperm(valid_indices.numel(), device=context_mask.device)
        ]
        drop_mask[row_idx, shuffled[:n_drop]] = True
    return drop_mask


def repeat_batch_tensor(
    tensor: torch.Tensor,
    repeat_count: int,
) -> torch.Tensor:
    if repeat_count == 1:
        return tensor.clone()
    repeat_shape = (repeat_count,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(repeat_shape)


def apply_token_drop(
    batch_data,
    drop_mask: torch.Tensor,
    pad_token_id: int,
    repeat_count: int = 1,
) -> dict[str, torch.Tensor]:
    input_ids = repeat_batch_tensor(
        tensor=batch_data.input_ids,
        repeat_count=repeat_count,
    )
    attention_mask = repeat_batch_tensor(
        tensor=batch_data.input_masks,
        repeat_count=repeat_count,
    )
    gaze_features = repeat_batch_tensor(
        tensor=batch_data.eyes,
        repeat_count=repeat_count,
    )
    claim_token_masks = getattr(
        batch_data,
        'claim_token_masks',
        torch.zeros_like(attention_mask),
    )
    if claim_token_masks.shape[0] != attention_mask.shape[0]:
        claim_token_masks = repeat_batch_tensor(
            tensor=claim_token_masks,
            repeat_count=repeat_count,
        )
    else:
        claim_token_masks = claim_token_masks.clone()

    context_token_masks = getattr(
        batch_data,
        'context_token_masks',
        attention_mask.clone(),
    )
    if context_token_masks.shape[0] != attention_mask.shape[0]:
        context_token_masks = repeat_batch_tensor(
            tensor=context_token_masks,
            repeat_count=repeat_count,
        )
    else:
        context_token_masks = context_token_masks.clone()

    if not hasattr(batch_data, 'context_token_masks'):
        context_token_masks[:, 0] = 0

    input_ids[drop_mask] = pad_token_id
    attention_mask[drop_mask] = 0
    claim_token_masks[drop_mask] = 0
    context_token_masks[drop_mask] = 0
    gaze_features[drop_mask] = 0

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gaze_features': gaze_features,
        'claim_token_masks': claim_token_masks,
        'context_token_masks': context_token_masks,
    }


def forward_probs(
    model: CECGazeModel,
    batch_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    gold_labels: torch.Tensor,
    annotator_labels: torch.Tensor,
) -> torch.Tensor:
    output = model(
        input_ids=batch_inputs['input_ids'],
        attention_mask=batch_inputs['attention_mask'],
        gaze_features=batch_inputs['gaze_features'],
        claim_token_masks=batch_inputs['claim_token_masks'],
        context_token_masks=batch_inputs['context_token_masks'],
        labels=labels,
        gold_labels=gold_labels,
        annotator_labels=annotator_labels,
    )
    return get_positive_class_probs(output.logits)


def run_score_drop_batch(
    model: CECGazeModel,
    batch: list,
    device: torch.device,
    drop_fraction: float,
    random_repeats: int,
) -> ScoreDropBatchResult:
    batch = move_batch_to_device(batch=batch, device=device)
    batch_data = model.unpack_batch(batch)

    labels = batch_data.labels
    gold_labels = (
        batch_data.gold_labels if hasattr(batch_data, 'gold_labels') else labels
    )
    annotator_labels = (
        batch_data.annotator_labels
        if hasattr(batch_data, 'annotator_labels')
        else labels
    )
    attention_mask = batch_data.input_masks
    context_mask = get_context_mask(batch_data=batch_data, attention_mask=attention_mask)
    pad_token_id = getattr(model.encoder.config, 'pad_token_id', 1) or 1

    base_output = model(
        input_ids=batch_data.input_ids,
        attention_mask=attention_mask,
        gaze_features=batch_data.eyes,
        claim_token_masks=getattr(batch_data, 'claim_token_masks', None),
        context_token_masks=getattr(batch_data, 'context_token_masks', None),
        labels=labels,
        gold_labels=gold_labels,
        annotator_labels=annotator_labels,
    )
    base_probs = get_positive_class_probs(base_output.logits)

    top_drop_mask, removed_top_mass, num_dropped_tokens = build_top_drop_mask(
        evidence_weights=base_output.evidence_weights,
        context_mask=context_mask,
        drop_fraction=drop_fraction,
    )
    top_drop_inputs = apply_token_drop(
        batch_data=batch_data,
        drop_mask=top_drop_mask,
        pad_token_id=pad_token_id,
    )
    top_drop_probs = forward_probs(
        model=model,
        batch_inputs=top_drop_inputs,
        labels=labels,
        gold_labels=gold_labels,
        annotator_labels=annotator_labels,
    )

    random_probs = []
    random_context_mask = repeat_batch_tensor(
        tensor=context_mask,
        repeat_count=random_repeats,
    )
    random_num_dropped_tokens = repeat_batch_tensor(
        tensor=num_dropped_tokens,
        repeat_count=random_repeats,
    )
    random_drop_mask = build_random_drop_mask(
        context_mask=random_context_mask,
        num_dropped_tokens=random_num_dropped_tokens,
    )
    random_drop_inputs = apply_token_drop(
        batch_data=batch_data,
        drop_mask=random_drop_mask,
        pad_token_id=pad_token_id,
        repeat_count=random_repeats,
    )
    random_probs_tensor = forward_probs(
        model=model,
        batch_inputs=random_drop_inputs,
        labels=repeat_batch_tensor(
            tensor=labels,
            repeat_count=random_repeats,
        ),
        gold_labels=repeat_batch_tensor(
            tensor=gold_labels,
            repeat_count=random_repeats,
        ),
        annotator_labels=repeat_batch_tensor(
            tensor=annotator_labels,
            repeat_count=random_repeats,
        )
    ).view(random_repeats, labels.size(0))
    random_drop_probs_mean = random_probs_tensor.mean(dim=0)
    random_drop_probs_std = random_probs_tensor.std(dim=0, unbiased=False)

    return ScoreDropBatchResult(
        labels=get_class_labels(labels=labels).detach().cpu(),
        base_probs=base_probs.detach().cpu(),
        top_drop_probs=top_drop_probs.detach().cpu(),
        random_drop_probs_mean=random_drop_probs_mean.detach().cpu(),
        random_drop_probs_std=random_drop_probs_std.detach().cpu(),
        top_drop_delta=(base_probs - top_drop_probs).detach().cpu(),
        random_drop_delta_mean=(base_probs - random_drop_probs_mean).detach().cpu(),
        removed_top_mass=removed_top_mass.detach().cpu(),
        num_dropped_tokens=num_dropped_tokens.detach().cpu(),
    )


def evaluate_dataset(
    model: CECGazeModel,
    dataset,
    dataloader,
    eval_type: str,
    eval_regime: str,
    fold_index: int,
    device: torch.device,
    drop_fraction: float,
    random_repeats: int,
) -> pd.DataFrame:
    batch_results = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f'Score-drop {eval_type} {eval_regime} fold {fold_index}',
        ):
            batch_results.append(
                run_score_drop_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    drop_fraction=drop_fraction,
                    random_repeats=random_repeats,
                )
            )

    result_df = pd.DataFrame(
        {
            'label': torch.cat([res.labels for res in batch_results], dim=0).numpy(),
            'prediction_prob_base': torch.cat(
                [res.base_probs for res in batch_results], dim=0
            ).numpy(),
            'prediction_prob_top_drop': torch.cat(
                [res.top_drop_probs for res in batch_results], dim=0
            ).numpy(),
            'prediction_prob_random_drop_mean': torch.cat(
                [res.random_drop_probs_mean for res in batch_results], dim=0
            ).numpy(),
            'prediction_prob_random_drop_std': torch.cat(
                [res.random_drop_probs_std for res in batch_results], dim=0
            ).numpy(),
            'prediction_prob_delta_top_drop': torch.cat(
                [res.top_drop_delta for res in batch_results], dim=0
            ).numpy(),
            'prediction_prob_delta_random_drop_mean': torch.cat(
                [res.random_drop_delta_mean for res in batch_results], dim=0
            ).numpy(),
            'removed_top_evidence_mass': torch.cat(
                [res.removed_top_mass for res in batch_results], dim=0
            ).numpy(),
            'n_dropped_tokens': torch.cat(
                [res.num_dropped_tokens for res in batch_results], dim=0
            ).numpy(),
            'eval_regime': eval_regime,
            'eval_type': eval_type,
            'fold_index': fold_index,
        }
    )
    trial_info = extract_trial_info(
        dataset,
        cols_to_keep=dataset.trial_groupby_columns,
    ).reset_index(drop=True)
    trial_info = trial_info.drop(
        columns=[col for col in trial_info.columns if col in result_df.columns],
        errors='ignore',
    )
    return pd.concat([result_df, trial_info], axis=1)


def summarize_score_drop_results(results_df: pd.DataFrame) -> pd.DataFrame:
    def as_flat_array(values) -> np.ndarray:
        return np.asarray(values).reshape(-1)

    def safe_auroc(labels, probs) -> float:
        labels = as_flat_array(labels)
        probs = as_flat_array(probs)
        if len(pd.unique(labels)) < 2:
            return float('nan')
        return roc_auc_score(labels, probs)

    summary_rows = []
    for (eval_type, eval_regime, fold_index), group_df in results_df.groupby(
        ['eval_type', 'eval_regime', 'fold_index']
    ):
        labels = as_flat_array(group_df['label'].to_numpy())
        base_probs = as_flat_array(group_df['prediction_prob_base'].to_numpy())
        top_probs = as_flat_array(group_df['prediction_prob_top_drop'].to_numpy())
        random_probs = as_flat_array(
            group_df['prediction_prob_random_drop_mean'].to_numpy()
        )

        summary_rows.append(
            {
                'eval_type': eval_type,
                'eval_regime': eval_regime,
                'fold_index': fold_index,
                'mean_abs_delta_top_drop': group_df[
                    'prediction_prob_delta_top_drop'
                ].abs().mean(),
                'mean_abs_delta_random_drop': group_df[
                    'prediction_prob_delta_random_drop_mean'
                ].abs().mean(),
                'mean_signed_delta_top_drop': group_df[
                    'prediction_prob_delta_top_drop'
                ].mean(),
                'mean_signed_delta_random_drop': group_df[
                    'prediction_prob_delta_random_drop_mean'
                ].mean(),
                'mean_removed_top_evidence_mass': group_df[
                    'removed_top_evidence_mass'
                ].mean(),
                'mean_n_dropped_tokens': group_df['n_dropped_tokens'].mean(),
                'auroc_base': safe_auroc(labels, base_probs),
                'auroc_top_drop': safe_auroc(labels, top_probs),
                'auroc_random_drop': safe_auroc(labels, random_probs),
                'balanced_accuracy_base': balanced_accuracy_score(
                    labels,
                    (base_probs >= 0.5).astype(int),
                ),
                'balanced_accuracy_top_drop': balanced_accuracy_score(
                    labels,
                    (top_probs >= 0.5).astype(int),
                ),
                'balanced_accuracy_random_drop': balanced_accuracy_score(
                    labels,
                    (random_probs >= 0.5).astype(int),
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    lf.seed_everything(args.seed, workers=True, verbose=False)
    torch.set_float32_matmul_precision('high')

    fold_paths = get_fold_paths(args.eval_path)
    if not fold_paths:
        raise FileNotFoundError(f'No fold_index=* directories found in {args.eval_path}')

    device = choose_device(device_name=args.device)
    logger.info(f'Running score-drop evaluation on {device}')

    for fold_path in fold_paths:
        fold_index = int(fold_path.name.split('=')[1])
        try:
            cfg, model = load_cec_gaze_model(
                fold_path=fold_path,
                checkpoint_template=args.checkpoint_template,
                error_context='Score-drop evaluation',
            )
        except (AssertionError, FileNotFoundError, TypeError) as exc:
            logger.warning(f'Skipping fold {fold_index}! {exc}')
            continue

        model = model.to(device)
        if args.batch_size is not None:
            cfg.model.batch_size = args.batch_size
        if args.num_workers is not None:
            cfg.trainer.num_workers = args.num_workers
        dm = DataModuleFactory.get(datamodule_name=cfg.data.datamodule_name)(cfg)
        dm.prepare_data()
        dm.setup(stage='predict')
        val_dataloaders = dm.val_dataloader()
        test_dataloaders = dm.test_dataloader()

        fold_results = []
        for idx, regime_name in enumerate(REGIMES):
            if 'val' in args.eval_types:
                fold_results.append(
                    evaluate_dataset(
                        model=model,
                        dataset=dm.val_datasets[idx],
                        dataloader=val_dataloaders[idx],
                        eval_type='val',
                        eval_regime=regime_name,
                        fold_index=fold_index,
                        device=device,
                        drop_fraction=args.drop_fraction,
                        random_repeats=args.random_repeats,
                    )
                )
            if 'test' in args.eval_types:
                fold_results.append(
                    evaluate_dataset(
                        model=model,
                        dataset=dm.test_datasets[idx],
                        dataloader=test_dataloaders[idx],
                        eval_type='test',
                        eval_regime=regime_name,
                        fold_index=fold_index,
                        device=device,
                        drop_fraction=args.drop_fraction,
                        random_repeats=args.random_repeats,
                    )
                )

        fold_df = pd.concat(fold_results, ignore_index=True)
        summary_df = summarize_score_drop_results(results_df=fold_df)

        results_path = fold_path / args.output_name
        summary_path = fold_path / args.summary_name
        fold_df.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f'Saved score-drop rows to {results_path}')
        logger.info(f'Saved score-drop summary to {summary_path}')


if __name__ == '__main__':
    main()
