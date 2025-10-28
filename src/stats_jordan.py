"""Module for statistical functions."""

from typing import NamedTuple  # noqa: I001
import pandas as pd

class Fences(NamedTuple):
    """Named tuple for Tukey's fences."""

    outer_lower: float | None
    inner_lower: float | None
    inner_upper: float | None
    outer_upper: float | None


def tukey_fences(feature_series: pd.Series) -> Fences:
    """Calculate and return Tukey's Fences of a pandas Series.

    The rounding precision is determined by the maximum number of decimal
    places in the input data.
    """
    clean_series = feature_series.dropna()

    if clean_series.empty:
        return Fences(outer_lower=None, inner_lower=None,
                      inner_upper=None, outer_upper=None)

    str_clean_series = clean_series.astype(str)

    if str_clean_series.str.contains(".", regex=False).any():
        round_num: int = str_clean_series.str.split(".", expand=True)[1].str.len().max()
    else:
        round_num: int = 0

    q1, q3 = clean_series.quantile([0.25, 0.75])
    iqr: float = q3 - q1
    outer_upper_fence: float = round(q3 + 3 * iqr, round_num)
    inner_upper_fence: float = round(q3 + 1.5 * iqr, round_num)
    inner_lower_fence: float = round(q1 - 1.5 * iqr, round_num)
    outer_lower_fence: float = round(q1 - 3 * iqr, round_num)

    return Fences(
        outer_lower=outer_lower_fence,
        inner_lower=inner_lower_fence,
        inner_upper=inner_upper_fence,
        outer_upper=outer_upper_fence
    )
