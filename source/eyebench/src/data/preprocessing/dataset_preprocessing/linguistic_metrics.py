"""Compatibility wrapper for linguistic metric and surprisal extractor imports."""

try:
    from psycholing_metrics import get_metrics
    from psycholing_metrics.text_processing import is_content_word
    from psycholing_metrics.surprisal.factory import (
        create_surprisal_extractor as get_surp_extractor,
    )
    from psycholing_metrics.surprisal.types import (
        SurprisalExtractorType as SurpExtractorType,
    )
except ModuleNotFoundError:
    from text_metrics.ling_metrics_funcs import get_metrics
    from text_metrics.utils import is_content_word
    from text_metrics.surprisal_extractors.extractor_switch import (
        get_surp_extractor,
    )
    from text_metrics.surprisal_extractors.extractors_constants import (
        SurpExtractorType,
    )
