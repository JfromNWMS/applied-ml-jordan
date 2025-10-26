"""A module for calculating Tukey's fences."""

from typing import NamedTuple  # noqa: I001
import pandas as pd

class TukeyFences(NamedTuple):
    """A named tuple to store the four Tukey fence values."""

    outer_lower: float
    inner_lower: float
    inner_upper: float
    outer_upper: float

def tukey_fences(feature_series: pd.Series) -> TukeyFences:
    """Return Tukey's Fences of a pandas Series."""
    q1, q3 = feature_series.quantile([0.25, 0.75])
    iqr: float = q3 - q1
    outer_upper_fence: float = q3 + 3 * iqr
    inner_upper_fence: float = q3 + 1.5 * iqr
    inner_lower_fence: float = q1 - 1.5 * iqr
    outer_lower_fence: float = q1 - 3 * iqr

    return TukeyFences(
        outer_lower=outer_lower_fence,
        inner_lower=inner_lower_fence,
        inner_upper=inner_upper_fence,
        outer_upper=outer_upper_fence
    )