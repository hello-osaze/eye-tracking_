from __future__ import annotations

from typing import Any

import pandas as pd


def safe_to_markdown(
    df: pd.DataFrame,
    *,
    index: bool = False,
    floatfmt: str | None = None,
    **kwargs: Any,
) -> str:
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt, **kwargs)
    except ImportError:
        return df.to_string(index=index, float_format=_make_float_format(floatfmt))


def _make_float_format(floatfmt: str | None):
    if not floatfmt:
        return None
    spec = floatfmt
    if spec.startswith('%'):
        def formatter(value: float) -> str:
            return spec % value
        return formatter
    def formatter(value: float) -> str:
        return format(value, spec)
    return formatter
