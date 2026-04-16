"""Evaluate explanation faithfulness for CEC-Gaze evidence scores.

This script measures both comprehensiveness and sufficiency using the learned
claim-conditioned evidence weights:

- comprehensiveness: how much the base predicted-class confidence drops when
  top-ranked context tokens are removed
- sufficiency: how much of the base predicted-class confidence remains when
  only the top-ranked context tokens are kept

For both notions we compare the learned ranking against random and bottom-ranked
controls across several token fractions.
"""

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
class FaithfulnessBatchResult:
    labels: torch.Tensor
    base_pred_class: torch.Tensor
    base_pred_conf: torch.Tensor
    comprehensiveness_top: torch.Tensor
    comprehensiveness_bottom: torch.Tensor
    comprehensiveness_random_mean: torch.Tensor
    comprehensiveness_random_std: torch.Tensor
    sufficiency_top_gap: torch.Tensor
    sufficiency_bottom_gap: torch.Tensor
    sufficiency_random_gap_mean: torch.Tensor
    sufficiency_random_gap_std: torch.Tensor
    top_drop_pred_conf: torch.Tensor
    bottom_drop_pred_conf: torch.Tensor
    random_drop_pred_conf_mean: torch.Tensor
    top_keep_pred_conf: torch.Tensor
    bottom_keep_pred_conf: torch.Tensor
    random_keep_pred_conf_mean: torch.Tensor
    removed_top_mass: torch.Tensor
    kept_top_mass: torch.Tensor
    num_selected_tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run comprehensiveness/sufficiency faithfulness tests for CEC-Gaze.'
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
        '--fractions',
        nargs='+',
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        help='Fractions of valid context tokens to keep/drop for faithfulness curves.',
    )
    parser.add_argument(
        '--random-repeats',
        default=10,
        type=int,
        help='Number of random baselines per perturbation.',
    )
    parser.add_argument(
        '--batch-size',
        default=None,
        type=int,
        help='Optional evaluation batch size override.',
    )
    parser.add_argument(
        '--eval-types',
        nargs='+',
        default=['test'],
        choices=['val', 'test'],
        help='Which dataset splits to evaluate.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        help='Device to use for evaluation.',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='Random seed.',
    )
    parser.add_argument(
        '--output-name',
        default='faithfulness_test_results.csv',
        help='Filename for per-trial faithfulness outputs saved under each fold.',
    )
    parser.add_argument(
        '--summary-name',
        default='faithfulness_summary.csv',
        help='Filename for grouped faithfulness summaries saved under each fold.',
    )
    args = parser.parse_args()
    for fraction in args.fractions:
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                f'Every --fractions value must be in (0, 1], got {fraction}.'
            )
    if args.random_repeats < 1:
        raise ValueError(
            f'--random-repeats must be >= 1, got {args.random_repeats}.'
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f'--batch-size must be >= 1, got {args.batch_size}.')
    return args


def get_context_mask(batch_data, attention_mask: torch.Tensor) -> torch.Tensor:
    context_token_masks = getattr(batch_data, 'context_token_masks', None)
    if context_token_masks is None:
        context_token_masks = attention_mask.clone()
        context_token_masks[:, 0] = 0
    return context_token_masks.bool() & attention_mask.bool()


def get_class_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1 and labels.shape[-1] > 1:
        labels = labels.argmax(dim=-1)
    return labels.reshape(-1)


def choose_num_selected_tokens(valid_count: int, fraction: float) -> int:
    if valid_count <= 0:
        return 0
    return max(1, min(valid_count, int(math.ceil(valid_count * fraction))))


def repeat_batch_tensor(
    tensor: torch.Tensor,
    repeat_count: int,
) -> torch.Tensor:
    if repeat_count == 1:
        return tensor.clone()
    repeat_shape = (repeat_count,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(repeat_shape)


def get_predicted_class_confidence(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(dim=1)
    pred_class = probs.argmax(dim=1)
    pred_conf = probs.gather(dim=1, index=pred_class.unsqueeze(1)).squeeze(1)
    return pred_class, pred_conf


def get_fixed_class_confidence(
    logits: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(dim=1)
    return probs.gather(dim=1, index=target_classes.unsqueeze(1)).squeeze(1)


def build_selection_masks(
    evidence_weights: torch.Tensor,
    context_mask: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top_keep_mask = torch.zeros_like(context_mask)
    bottom_keep_mask = torch.zeros_like(context_mask)
    kept_top_mass = torch.zeros(
        evidence_weights.size(0),
        device=evidence_weights.device,
        dtype=evidence_weights.dtype,
    )
    num_selected_tokens = torch.zeros(
        evidence_weights.size(0),
        device=evidence_weights.device,
        dtype=torch.long,
    )

    for row_idx in range(evidence_weights.size(0)):
        valid_indices = torch.nonzero(context_mask[row_idx], as_tuple=False).squeeze(-1)
        n_select = choose_num_selected_tokens(
            valid_count=valid_indices.numel(),
            fraction=fraction,
        )
        if n_select == 0:
            continue
        valid_weights = evidence_weights[row_idx, valid_indices]
        top_order = torch.topk(valid_weights, k=n_select).indices
        bottom_order = torch.topk(valid_weights, k=n_select, largest=False).indices
        top_indices = valid_indices[top_order]
        bottom_indices = valid_indices[bottom_order]

        top_keep_mask[row_idx, top_indices] = True
        bottom_keep_mask[row_idx, bottom_indices] = True
        kept_top_mass[row_idx] = evidence_weights[row_idx, top_indices].sum()
        num_selected_tokens[row_idx] = n_select

    # For comprehensiveness, we drop the selected tokens and keep the rest.
    top_drop_mask = top_keep_mask.clone()
    bottom_drop_mask = bottom_keep_mask.clone()
    return top_keep_mask, bottom_keep_mask, top_drop_mask, bottom_drop_mask, kept_top_mass, num_selected_tokens


def build_random_keep_mask(
    context_mask: torch.Tensor,
    num_selected_tokens: torch.Tensor,
) -> torch.Tensor:
    keep_mask = torch.zeros_like(context_mask)
    for row_idx in range(context_mask.size(0)):
        valid_indices = torch.nonzero(context_mask[row_idx], as_tuple=False).squeeze(-1)
        n_select = int(num_selected_tokens[row_idx].item())
        if n_select <= 0 or valid_indices.numel() == 0:
            continue
        n_select = min(n_select, valid_indices.numel())
        shuffled = valid_indices[
            torch.randperm(valid_indices.numel(), device=context_mask.device)
        ]
        keep_mask[row_idx, shuffled[:n_select]] = True
    return keep_mask


def apply_context_perturbation(
    batch_data,
    context_keep_mask: torch.Tensor,
    context_drop_mask: torch.Tensor,
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

    if context_drop_mask.any():
        input_ids[context_drop_mask] = pad_token_id
        attention_mask[context_drop_mask] = 0
        claim_token_masks[context_drop_mask] = 0
        context_token_masks[context_drop_mask] = 0
        gaze_features[context_drop_mask] = 0

    if context_keep_mask.any():
        context_keep_mask = context_keep_mask & context_token_masks.bool()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gaze_features': gaze_features,
        'claim_token_masks': claim_token_masks,
        'context_token_masks': context_token_masks,
    }


def forward_logits(
    model: CECGazeModel,
    batch_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    gold_labels: torch.Tensor,
    annotator_labels: torch.Tensor,
):
    return model(
        input_ids=batch_inputs['input_ids'],
        attention_mask=batch_inputs['attention_mask'],
        gaze_features=batch_inputs['gaze_features'],
        claim_token_masks=batch_inputs['claim_token_masks'],
        context_token_masks=batch_inputs['context_token_masks'],
        labels=labels,
        gold_labels=gold_labels,
        annotator_labels=annotator_labels,
    )


def run_faithfulness_batch(
    model: CECGazeModel,
    batch: list,
    device: torch.device,
    fraction: float,
    random_repeats: int,
) -> FaithfulnessBatchResult:
    batch = move_batch_to_device(batch=batch, device=device)
    batch_data = model.unpack_batch(batch)

    labels = batch_data.labels
    gold_labels = (
        batch_data.gold_labels if hasattr(batch_data, 'gold_labels') else labels
    )
    annotator_labels = (
        batch_data.annotator_labels if hasattr(batch_data, 'annotator_labels') else labels
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
    base_pred_class, base_pred_conf = get_predicted_class_confidence(base_output.logits)

    (
        top_keep_mask,
        bottom_keep_mask,
        top_drop_mask,
        bottom_drop_mask,
        kept_top_mass,
        num_selected_tokens,
    ) = build_selection_masks(
        evidence_weights=base_output.evidence_weights,
        context_mask=context_mask,
        fraction=fraction,
    )
    removed_top_mass = (
        base_output.evidence_weights.masked_fill(~top_keep_mask, 0.0).sum(dim=1)
    )

    top_drop_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=context_mask & ~top_drop_mask,
        context_drop_mask=top_drop_mask,
        pad_token_id=pad_token_id,
    )
    bottom_drop_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=context_mask & ~bottom_drop_mask,
        context_drop_mask=bottom_drop_mask,
        pad_token_id=pad_token_id,
    )

    top_keep_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=top_keep_mask,
        context_drop_mask=context_mask & ~top_keep_mask,
        pad_token_id=pad_token_id,
    )
    bottom_keep_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=bottom_keep_mask,
        context_drop_mask=context_mask & ~bottom_keep_mask,
        pad_token_id=pad_token_id,
    )

    top_drop_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=top_drop_inputs,
            labels=labels,
            gold_labels=gold_labels,
            annotator_labels=annotator_labels,
        ).logits,
        target_classes=base_pred_class,
    )
    bottom_drop_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=bottom_drop_inputs,
            labels=labels,
            gold_labels=gold_labels,
            annotator_labels=annotator_labels,
        ).logits,
        target_classes=base_pred_class,
    )
    top_keep_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=top_keep_inputs,
            labels=labels,
            gold_labels=gold_labels,
            annotator_labels=annotator_labels,
        ).logits,
        target_classes=base_pred_class,
    )
    bottom_keep_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=bottom_keep_inputs,
            labels=labels,
            gold_labels=gold_labels,
            annotator_labels=annotator_labels,
        ).logits,
        target_classes=base_pred_class,
    )

    random_context_mask = repeat_batch_tensor(
        tensor=context_mask,
        repeat_count=random_repeats,
    )
    random_num_selected = repeat_batch_tensor(
        tensor=num_selected_tokens,
        repeat_count=random_repeats,
    )
    random_keep_mask = build_random_keep_mask(
        context_mask=random_context_mask,
        num_selected_tokens=random_num_selected,
    )
    random_drop_mask = random_keep_mask
    random_keep_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=random_keep_mask,
        context_drop_mask=random_context_mask & ~random_keep_mask,
        pad_token_id=pad_token_id,
        repeat_count=random_repeats,
    )
    random_drop_inputs = apply_context_perturbation(
        batch_data=batch_data,
        context_keep_mask=random_context_mask & ~random_drop_mask,
        context_drop_mask=random_drop_mask,
        pad_token_id=pad_token_id,
        repeat_count=random_repeats,
    )

    repeated_base_classes = repeat_batch_tensor(
        tensor=base_pred_class,
        repeat_count=random_repeats,
    )
    repeated_labels = repeat_batch_tensor(tensor=labels, repeat_count=random_repeats)
    repeated_gold_labels = repeat_batch_tensor(
        tensor=gold_labels,
        repeat_count=random_repeats,
    )
    repeated_annotator_labels = repeat_batch_tensor(
        tensor=annotator_labels,
        repeat_count=random_repeats,
    )

    random_keep_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=random_keep_inputs,
            labels=repeated_labels,
            gold_labels=repeated_gold_labels,
            annotator_labels=repeated_annotator_labels,
        ).logits,
        target_classes=repeated_base_classes,
    ).view(random_repeats, labels.size(0))
    random_drop_conf = get_fixed_class_confidence(
        forward_logits(
            model=model,
            batch_inputs=random_drop_inputs,
            labels=repeated_labels,
            gold_labels=repeated_gold_labels,
            annotator_labels=repeated_annotator_labels,
        ).logits,
        target_classes=repeated_base_classes,
    ).view(random_repeats, labels.size(0))

    random_keep_conf_mean = random_keep_conf.mean(dim=0)
    random_keep_conf_std = random_keep_conf.std(dim=0, unbiased=False)
    random_drop_conf_mean = random_drop_conf.mean(dim=0)
    random_drop_conf_std = random_drop_conf.std(dim=0, unbiased=False)

    return FaithfulnessBatchResult(
        labels=get_class_labels(labels=labels).detach().cpu(),
        base_pred_class=base_pred_class.detach().cpu(),
        base_pred_conf=base_pred_conf.detach().cpu(),
        comprehensiveness_top=(base_pred_conf - top_drop_conf).detach().cpu(),
        comprehensiveness_bottom=(base_pred_conf - bottom_drop_conf).detach().cpu(),
        comprehensiveness_random_mean=(base_pred_conf - random_drop_conf_mean).detach().cpu(),
        comprehensiveness_random_std=random_drop_conf_std.detach().cpu(),
        sufficiency_top_gap=(base_pred_conf - top_keep_conf).detach().cpu(),
        sufficiency_bottom_gap=(base_pred_conf - bottom_keep_conf).detach().cpu(),
        sufficiency_random_gap_mean=(base_pred_conf - random_keep_conf_mean).detach().cpu(),
        sufficiency_random_gap_std=random_keep_conf_std.detach().cpu(),
        top_drop_pred_conf=top_drop_conf.detach().cpu(),
        bottom_drop_pred_conf=bottom_drop_conf.detach().cpu(),
        random_drop_pred_conf_mean=random_drop_conf_mean.detach().cpu(),
        top_keep_pred_conf=top_keep_conf.detach().cpu(),
        bottom_keep_pred_conf=bottom_keep_conf.detach().cpu(),
        random_keep_pred_conf_mean=random_keep_conf_mean.detach().cpu(),
        removed_top_mass=removed_top_mass.detach().cpu(),
        kept_top_mass=kept_top_mass.detach().cpu(),
        num_selected_tokens=num_selected_tokens.detach().cpu(),
    )


def evaluate_dataset_fraction(
    model: CECGazeModel,
    dataset,
    dataloader,
    eval_type: str,
    eval_regime: str,
    fold_index: int,
    device: torch.device,
    fraction: float,
    random_repeats: int,
) -> pd.DataFrame:
    batch_results = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f'Faithfulness {eval_type} {eval_regime} fold {fold_index} frac {fraction:.2f}',
        ):
            batch_results.append(
                run_faithfulness_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    fraction=fraction,
                    random_repeats=random_repeats,
                )
            )

    result_df = pd.DataFrame(
        {
            'label': torch.cat([res.labels for res in batch_results], dim=0).numpy(),
            'base_pred_class': torch.cat(
                [res.base_pred_class for res in batch_results], dim=0
            ).numpy(),
            'base_pred_conf': torch.cat(
                [res.base_pred_conf for res in batch_results], dim=0
            ).numpy(),
            'fraction': fraction,
            'comprehensiveness_top': torch.cat(
                [res.comprehensiveness_top for res in batch_results], dim=0
            ).numpy(),
            'comprehensiveness_bottom': torch.cat(
                [res.comprehensiveness_bottom for res in batch_results], dim=0
            ).numpy(),
            'comprehensiveness_random_mean': torch.cat(
                [res.comprehensiveness_random_mean for res in batch_results], dim=0
            ).numpy(),
            'comprehensiveness_random_std': torch.cat(
                [res.comprehensiveness_random_std for res in batch_results], dim=0
            ).numpy(),
            'sufficiency_top_gap': torch.cat(
                [res.sufficiency_top_gap for res in batch_results], dim=0
            ).numpy(),
            'sufficiency_bottom_gap': torch.cat(
                [res.sufficiency_bottom_gap for res in batch_results], dim=0
            ).numpy(),
            'sufficiency_random_gap_mean': torch.cat(
                [res.sufficiency_random_gap_mean for res in batch_results], dim=0
            ).numpy(),
            'sufficiency_random_gap_std': torch.cat(
                [res.sufficiency_random_gap_std for res in batch_results], dim=0
            ).numpy(),
            'top_drop_pred_conf': torch.cat(
                [res.top_drop_pred_conf for res in batch_results], dim=0
            ).numpy(),
            'bottom_drop_pred_conf': torch.cat(
                [res.bottom_drop_pred_conf for res in batch_results], dim=0
            ).numpy(),
            'random_drop_pred_conf_mean': torch.cat(
                [res.random_drop_pred_conf_mean for res in batch_results], dim=0
            ).numpy(),
            'top_keep_pred_conf': torch.cat(
                [res.top_keep_pred_conf for res in batch_results], dim=0
            ).numpy(),
            'bottom_keep_pred_conf': torch.cat(
                [res.bottom_keep_pred_conf for res in batch_results], dim=0
            ).numpy(),
            'random_keep_pred_conf_mean': torch.cat(
                [res.random_keep_pred_conf_mean for res in batch_results], dim=0
            ).numpy(),
            'removed_top_evidence_mass': torch.cat(
                [res.removed_top_mass for res in batch_results], dim=0
            ).numpy(),
            'kept_top_evidence_mass': torch.cat(
                [res.kept_top_mass for res in batch_results], dim=0
            ).numpy(),
            'n_selected_tokens': torch.cat(
                [res.num_selected_tokens for res in batch_results], dim=0
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


def summarize_faithfulness_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (
        eval_type,
        eval_regime,
        fold_index,
        fraction,
    ), group_df in results_df.groupby(
        ['eval_type', 'eval_regime', 'fold_index', 'fraction']
    ):
        summary_rows.append(
            {
                'eval_type': eval_type,
                'eval_regime': eval_regime,
                'fold_index': fold_index,
                'fraction': fraction,
                'mean_comprehensiveness_top': group_df['comprehensiveness_top'].mean(),
                'mean_comprehensiveness_bottom': group_df['comprehensiveness_bottom'].mean(),
                'mean_comprehensiveness_random': group_df[
                    'comprehensiveness_random_mean'
                ].mean(),
                'mean_sufficiency_top_gap': group_df['sufficiency_top_gap'].mean(),
                'mean_sufficiency_bottom_gap': group_df['sufficiency_bottom_gap'].mean(),
                'mean_sufficiency_random_gap': group_df[
                    'sufficiency_random_gap_mean'
                ].mean(),
                'comp_top_minus_random': (
                    group_df['comprehensiveness_top']
                    - group_df['comprehensiveness_random_mean']
                ).mean(),
                'comp_top_minus_bottom': (
                    group_df['comprehensiveness_top']
                    - group_df['comprehensiveness_bottom']
                ).mean(),
                'suff_random_minus_top': (
                    group_df['sufficiency_random_gap_mean']
                    - group_df['sufficiency_top_gap']
                ).mean(),
                'suff_bottom_minus_top': (
                    group_df['sufficiency_bottom_gap']
                    - group_df['sufficiency_top_gap']
                ).mean(),
                'fraction_comp_top_gt_random': (
                    group_df['comprehensiveness_top']
                    > group_df['comprehensiveness_random_mean']
                ).mean(),
                'fraction_comp_top_gt_bottom': (
                    group_df['comprehensiveness_top']
                    > group_df['comprehensiveness_bottom']
                ).mean(),
                'fraction_suff_top_lt_random': (
                    group_df['sufficiency_top_gap']
                    < group_df['sufficiency_random_gap_mean']
                ).mean(),
                'fraction_suff_top_lt_bottom': (
                    group_df['sufficiency_top_gap']
                    < group_df['sufficiency_bottom_gap']
                ).mean(),
                'mean_removed_top_evidence_mass': group_df[
                    'removed_top_evidence_mass'
                ].mean(),
                'mean_kept_top_evidence_mass': group_df[
                    'kept_top_evidence_mass'
                ].mean(),
                'mean_n_selected_tokens': group_df['n_selected_tokens'].mean(),
                'n_trials': len(group_df),
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

    device = choose_device(args.device)
    logger.info(f'Running faithfulness evaluation on {device}')

    for fold_path in fold_paths:
        fold_index = int(fold_path.name.split('=')[1])
        try:
            cfg, model = load_cec_gaze_model(
                fold_path=fold_path,
                checkpoint_template=args.checkpoint_template,
                error_context='Faithfulness evaluation',
            )
        except (AssertionError, FileNotFoundError, TypeError) as exc:
            logger.warning(f'Skipping fold {fold_index}! {exc}')
            continue

        model = model.to(device)
        if args.batch_size is not None:
            cfg.model.batch_size = args.batch_size
        dm = DataModuleFactory.get(datamodule_name=cfg.data.datamodule_name)(cfg)
        dm.prepare_data()
        dm.setup(stage='predict')
        val_dataloaders = dm.val_dataloader()
        test_dataloaders = dm.test_dataloader()

        fold_results = []
        for fraction in args.fractions:
            for idx, regime_name in enumerate(REGIMES):
                if 'val' in args.eval_types:
                    fold_results.append(
                        evaluate_dataset_fraction(
                            model=model,
                            dataset=dm.val_datasets[idx],
                            dataloader=val_dataloaders[idx],
                            eval_type='val',
                            eval_regime=regime_name,
                            fold_index=fold_index,
                            device=device,
                            fraction=fraction,
                            random_repeats=args.random_repeats,
                        )
                    )
                if 'test' in args.eval_types:
                    fold_results.append(
                        evaluate_dataset_fraction(
                            model=model,
                            dataset=dm.test_datasets[idx],
                            dataloader=test_dataloaders[idx],
                            eval_type='test',
                            eval_regime=regime_name,
                            fold_index=fold_index,
                            device=device,
                            fraction=fraction,
                            random_repeats=args.random_repeats,
                        )
                    )

        fold_df = pd.concat(fold_results, ignore_index=True)
        summary_df = summarize_faithfulness_results(results_df=fold_df)

        results_path = fold_path / args.output_name
        summary_path = fold_path / args.summary_name
        fold_df.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f'Saved faithfulness rows to {results_path}')
        logger.info(f'Saved faithfulness summary to {summary_path}')


if __name__ == '__main__':
    main()
