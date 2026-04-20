from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.roberta import RobertaModel
from transformers.models.xlm_roberta import XLMRobertaModel
from transformers.optimization import get_linear_schedule_with_warmup

from src.configs.data import DataArgs
from src.configs.models.dl.CECGaze import CECGaze
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


@dataclass
class CECGazeForwardOutput:
    logits: torch.Tensor
    gold_logits: torch.Tensor
    annotator_logits: torch.Tensor
    evidence_weights: torch.Tensor
    coverage_vector: torch.Tensor
    main_loss: torch.Tensor | None = None
    gold_loss: torch.Tensor | None = None
    annotator_loss: torch.Tensor | None = None
    sparse_loss: torch.Tensor | None = None
    loss: torch.Tensor | None = None


@register_model
class CECGazeModel(BaseModel):
    """Claim-conditioned evidence coverage model with explicit ablation controls."""

    COVERAGE_FEATURES = (
        'IA_DWELL_TIME',
        'IA_FIXATION_COUNT',
        'IA_REGRESSION_IN_COUNT',
        'IA_SKIP',
    )
    EXTRA_REGRESSION_FEATURES = (
        'IA_REGRESSION_OUT_FULL_COUNT',
        'IA_REGRESSION_OUT_COUNT',
    )

    def __init__(
        self,
        model_args: CECGaze,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args,
            trainer_args=trainer_args,
            data_args=data_args,
        )
        if model_args.use_fixation_report:
            raise ValueError(
                'CECGazeModel currently expects word-level IA features, not fixation reports.'
            )
        self.model_args = model_args
        self.warmup_proportion = model_args.warmup_proportion
        self.fusion_hidden_size = model_args.fusion_hidden_size
        self.score_eval_mode = model_args.score_eval_mode
        if self.score_eval_mode not in {'learned', 'uniform', 'shuffle'}:
            raise ValueError(
                f'Invalid score_eval_mode={self.score_eval_mode}. '
                'Expected one of: learned, uniform, shuffle.'
            )
        self.use_evidence_scorer = model_args.use_evidence_scorer
        self.use_gaze_coverage = model_args.use_gaze_coverage
        self.use_gaze_features = model_args.use_gaze_features
        self.use_global_summary = model_args.use_global_summary
        self.eval_zero_coverage = model_args.eval_zero_coverage
        self.eval_zero_gaze_features = model_args.eval_zero_gaze_features
        self.lambda_gold = model_args.lambda_gold
        self.lambda_annotator = model_args.lambda_annotator
        self.lambda_sparse = model_args.lambda_sparse

        self.encoder = self._build_text_encoder(model_args=model_args, data_args=data_args)

        gaze_dim = model_args.eyes_dim if self.use_gaze_features else 0
        self.token_fusion = nn.Sequential(
            nn.Linear(model_args.text_dim + gaze_dim, model_args.fusion_hidden_size),
            nn.Tanh(),
            nn.Dropout(model_args.dropout_prob),
        )
        self.evidence_scorer = nn.Sequential(
            nn.Linear(
                model_args.fusion_hidden_size + model_args.text_dim,
                model_args.evidence_hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(model_args.dropout_prob),
            nn.Linear(model_args.evidence_hidden_size, 1),
        )
        self.coverage_projection = nn.Sequential(
            nn.Linear(len(self.COVERAGE_FEATURES), model_args.coverage_hidden_size),
            nn.ReLU(),
            nn.Dropout(model_args.dropout_prob),
        )

        corr_input_dim = (
            model_args.text_dim
            + model_args.fusion_hidden_size
            + model_args.coverage_hidden_size
        )
        gold_input_dim = model_args.text_dim + model_args.fusion_hidden_size
        if self.use_global_summary:
            corr_input_dim += model_args.text_dim
            gold_input_dim += model_args.text_dim

        def build_head(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(model_args.dropout_prob),
                nn.Linear(input_dim, model_args.head_hidden_size),
                nn.Tanh(),
                nn.Dropout(model_args.dropout_prob),
                nn.Linear(model_args.head_hidden_size, self.num_classes),
            )

        self.correctness_head = build_head(corr_input_dim)
        self.gold_head = build_head(gold_input_dim)
        self.annotator_head = build_head(corr_input_dim)

        numeric_ia_features = [
            feature
            for feature in model_args.ia_features
            if feature not in model_args.ia_categorical_features
        ]
        self.feature_indices = {
            feature_name: feature_idx
            for feature_idx, feature_name in enumerate(numeric_ia_features)
        }

        if model_args.freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.auxiliary_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.latest_loss_components: dict[str, torch.Tensor] = {}

        self.save_hyperparameters()

    @staticmethod
    def _build_text_encoder(model_args: CECGaze, data_args: DataArgs) -> nn.Module:
        if data_args.is_english:
            return RobertaModel.from_pretrained(model_args.backbone)
        return XLMRobertaModel.from_pretrained(model_args.backbone)

    @staticmethod
    def _masked_mean(
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        fallback_states: torch.Tensor,
    ) -> torch.Tensor:
        mask = token_mask.unsqueeze(dim=-1).to(dtype=token_states.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled = (token_states * mask).sum(dim=1) / denominator
        has_tokens = token_mask.sum(dim=1, keepdim=True) > 0
        return torch.where(has_tokens, pooled, fallback_states)

    @staticmethod
    def _ensure_non_empty_context_mask(
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_mask = context_mask & attention_mask.bool()
        empty_rows = safe_mask.sum(dim=1) == 0
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
        return safe_mask

    @staticmethod
    def _masked_uniform_weights(mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(dtype=torch.float32)
        return mask_float / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)

    @staticmethod
    def _shuffle_weights(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shuffled_weights = weights.clone()
        for row_idx in range(weights.size(0)):
            valid_indices = torch.nonzero(mask[row_idx], as_tuple=False).squeeze(dim=-1)
            if valid_indices.numel() <= 1:
                continue
            permutation = valid_indices[
                torch.randperm(valid_indices.numel(), device=weights.device)
            ]
            shuffled_weights[row_idx, valid_indices] = weights[row_idx, permutation]
        return shuffled_weights

    def _compute_evidence_weights(
        self,
        scorer_logits: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_evidence_scorer:
            return self._masked_uniform_weights(mask=context_mask)

        masked_logits = scorer_logits.masked_fill(~context_mask, -1e4)
        weights = torch.softmax(masked_logits, dim=1) * context_mask.to(
            dtype=masked_logits.dtype
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if not self.training and self.score_eval_mode == 'uniform':
            return self._masked_uniform_weights(mask=context_mask)
        if not self.training and self.score_eval_mode == 'shuffle':
            return self._shuffle_weights(weights=weights, mask=context_mask)
        return weights

    def _get_gaze_feature(
        self,
        gaze_features: torch.Tensor,
        feature_name: str,
    ) -> torch.Tensor:
        feature_idx = self.feature_indices.get(feature_name)
        if feature_idx is None or not self.use_gaze_features:
            return torch.zeros(
                gaze_features.shape[:2],
                device=gaze_features.device,
                dtype=gaze_features.dtype,
            )
        return gaze_features[:, :, feature_idx]

    def _compute_coverage_vector(
        self,
        gaze_features: torch.Tensor,
        evidence_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_gaze_coverage or not self.use_gaze_features:
            return torch.zeros(
                gaze_features.size(0),
                len(self.COVERAGE_FEATURES),
                device=gaze_features.device,
                dtype=gaze_features.dtype,
            )

        dwell_time = self._get_gaze_feature(gaze_features, 'IA_DWELL_TIME')
        fixation_count = self._get_gaze_feature(gaze_features, 'IA_FIXATION_COUNT')
        regression_activity = self._get_gaze_feature(
            gaze_features,
            'IA_REGRESSION_IN_COUNT',
        )
        for feature_name in self.EXTRA_REGRESSION_FEATURES:
            regression_activity = regression_activity + self._get_gaze_feature(
                gaze_features,
                feature_name,
            )
        skip_indicator = self._get_gaze_feature(gaze_features, 'IA_SKIP')
        visited_indicator = 1.0 - skip_indicator

        token_stats = torch.stack(
            [
                dwell_time,
                fixation_count,
                regression_activity,
                visited_indicator,
            ],
            dim=-1,
        )
        return (evidence_weights.unsqueeze(dim=-1) * token_stats).sum(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gaze_features: torch.Tensor,
        claim_token_masks: torch.Tensor | None = None,
        context_token_masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        gold_labels: torch.Tensor | None = None,
        annotator_labels: torch.Tensor | None = None,
    ) -> CECGazeForwardOutput:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_states = encoder_outputs.last_hidden_state

        if claim_token_masks is None:
            claim_token_masks = torch.zeros_like(attention_mask)
        if context_token_masks is None:
            context_token_masks = attention_mask.clone()
            context_token_masks[:, 0] = 0

        claim_mask = claim_token_masks.bool()
        context_mask = self._ensure_non_empty_context_mask(
            context_mask=context_token_masks.bool(),
            attention_mask=attention_mask,
        )
        claim_summary = self._masked_mean(
            token_states=token_states,
            token_mask=claim_mask,
            fallback_states=token_states[:, 0, :],
        )

        eval_gaze_features = gaze_features
        if not self.training and self.eval_zero_gaze_features:
            eval_gaze_features = torch.zeros_like(gaze_features)

        token_inputs = [token_states]
        if self.use_gaze_features:
            token_inputs.append(eval_gaze_features)
        fused_tokens = self.token_fusion(torch.cat(token_inputs, dim=-1))

        scorer_input = torch.cat(
            [
                fused_tokens,
                claim_summary.unsqueeze(dim=1).expand(-1, fused_tokens.size(1), -1),
            ],
            dim=-1,
        )
        scorer_logits = self.evidence_scorer(scorer_input).squeeze(dim=-1)
        evidence_weights = self._compute_evidence_weights(
            scorer_logits=scorer_logits,
            context_mask=context_mask,
        )

        evidence_summary = (
            evidence_weights.unsqueeze(dim=-1) * fused_tokens
        ).sum(dim=1)
        coverage_vector = self._compute_coverage_vector(
            gaze_features=eval_gaze_features,
            evidence_weights=evidence_weights,
        )
        if not self.training and self.eval_zero_coverage:
            coverage_vector = torch.zeros_like(coverage_vector)
        projected_coverage = self.coverage_projection(coverage_vector)

        prediction_parts = [claim_summary, evidence_summary, projected_coverage]
        gold_parts = [claim_summary, evidence_summary]
        if self.use_global_summary:
            global_summary = token_states[:, 0, :]
            prediction_parts.append(global_summary)
            gold_parts.append(global_summary)

        prediction_input = torch.cat(prediction_parts, dim=-1)
        gold_input = torch.cat(gold_parts, dim=-1)
        logits = self.correctness_head(prediction_input)
        gold_logits = self.gold_head(gold_input)
        annotator_logits = self.annotator_head(prediction_input)

        output = CECGazeForwardOutput(
            logits=logits,
            gold_logits=gold_logits,
            annotator_logits=annotator_logits,
            evidence_weights=evidence_weights,
            coverage_vector=coverage_vector,
        )

        if labels is None:
            return output

        output.main_loss = self.loss(logits, labels)

        if gold_labels is None:
            gold_labels = labels
        if annotator_labels is None:
            annotator_labels = labels

        output.gold_loss = self.auxiliary_loss(gold_logits, gold_labels)
        output.annotator_loss = self.auxiliary_loss(
            annotator_logits,
            annotator_labels,
        )
        output.sparse_loss = -(
            evidence_weights
            * torch.log(evidence_weights.clamp_min(1e-12))
        ).sum(dim=1).mean()
        output.loss = (
            output.main_loss
            + self.lambda_gold * output.gold_loss
            + self.lambda_annotator * output.annotator_loss
            + self.lambda_sparse * output.sparse_loss
        )
        return output

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        labels = batch_data.labels
        gold_labels = (
            batch_data.gold_labels
            if hasattr(batch_data, 'gold_labels')
            else labels
        )
        annotator_labels = (
            batch_data.annotator_labels
            if hasattr(batch_data, 'annotator_labels')
            else labels
        )

        output = self(
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.input_masks,
            gaze_features=batch_data.eyes,
            claim_token_masks=getattr(batch_data, 'claim_token_masks', None),
            context_token_masks=getattr(batch_data, 'context_token_masks', None),
            labels=labels,
            gold_labels=gold_labels,
            annotator_labels=annotator_labels,
        )
        assert output.loss is not None
        self.latest_loss_components = {
            'main': output.main_loss.detach() if output.main_loss is not None else output.loss.detach(),
            'gold': output.gold_loss.detach() if output.gold_loss is not None else output.loss.detach(),
            'annotator': output.annotator_loss.detach()
            if output.annotator_loss is not None
            else output.loss.detach(),
            'sparse': output.sparse_loss.detach()
            if output.sparse_loss is not None
            else output.loss.detach(),
        }
        return labels, output.loss, output.logits

    def log_loss(self, loss: torch.Tensor, step_type: str, dataloader_idx=0) -> None:
        super().log_loss(loss=loss, step_type=step_type, dataloader_idx=dataloader_idx)
        if step_type == 'train':
            suffix = 'train'
        else:
            suffix = f'{step_type}_{self.regime_names[dataloader_idx]}'

        for loss_name, loss_value in self.latest_loss_components.items():
            self.log(
                name=f'loss/{loss_name}_{suffix}',
                value=loss_value,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=self.batch_size,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        assert self.warmup_proportion is not None
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(stepping_batches * self.warmup_proportion),
            num_training_steps=stepping_batches,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
