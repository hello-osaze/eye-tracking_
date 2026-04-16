from dataclasses import dataclass

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class CECGaze(DLModelArgs):
    """Claim-conditioned evidence coverage model for IITBHGC claim verification."""

    base_model_name: DLModelNames = DLModelNames.CEC_GAZE_MODEL
    feature_cache_suffix: str = '_claim_context_v2'

    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = False
    prepend_eye_features_to_text: bool = False

    warmup_proportion: float = 0.1
    max_epochs: int = 10
    early_stopping_patience: int = 3
    freeze_backbone: bool = False

    fusion_hidden_size: int = 256
    evidence_hidden_size: int = 256
    coverage_hidden_size: int = 32
    head_hidden_size: int = 256
    dropout_prob: float = 0.1

    lambda_gold: float = 0.1
    lambda_annotator: float = 0.0
    lambda_sparse: float = 0.0

    use_global_summary: bool = True
    use_evidence_scorer: bool = True
    use_gaze_coverage: bool = True
    use_gaze_features: bool = True
    score_eval_mode: str = 'learned'
    eval_zero_coverage: bool = False
    eval_zero_gaze_features: bool = False


@register_model_config
@dataclass
class CECGazeNoScorer(CECGaze):
    """Ablation: replace claim-conditioned evidence scoring with uniform context weights."""

    use_evidence_scorer: bool = False
    lambda_sparse: float = 0.0


@register_model_config
@dataclass
class CECGazeNoCoverage(CECGaze):
    """Ablation: remove the evidence coverage vector from the prediction heads."""

    use_gaze_coverage: bool = False


@register_model_config
@dataclass
class CECGazeTextOnly(CECGaze):
    """Ablation: keep the claim-conditioned scorer but remove gaze features."""

    use_gaze_features: bool = False
    use_gaze_coverage: bool = False
    lambda_annotator: float = 0.0
