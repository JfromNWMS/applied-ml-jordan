"""Module for statistical functions."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple


class Fences(NamedTuple):
    """Named tuple for Tukey's fences."""

    outer_lower: float | None
    inner_lower: float | None
    inner_upper: float | None
    outer_upper: float | None


def tukey_fences(feature_series: pd.Series, adjusted: bool = False) -> Fences:
    """Calculate and return Tukey's Fences of a pandas Series.

    The rounding precision is determined by the maximum number of decimal
    places in the input data.

    Parameters
    ----------
    feature_series : pd.Series
        The pandas Series for which to calculate the fences.
    adjusted : bool, default False
        If True, use the Medcouple adjustment for skewed distributions.
        This method was proposed by Hubert and Vandervieren in their paper
        "An adjusted boxplot for skewed distributions" (2008), *Computational
        Statistics and Data Analysis*.

    Returns
    -------
    Fences
        A NamedTuple containing the inner and outer, lower and upper fence values.
    """
    clean_series = feature_series.dropna()

    if clean_series.empty:
        return Fences(outer_lower=None, inner_lower=None,
                           inner_upper=None, outer_upper=None)

    if adjusted:
        mc: float = medcouple(clean_series.to_numpy())
        lower_multiplier: float = np.exp(-3.5 * mc).item()
        upper_multiplier: float = np.exp(4 * mc).item()
    else:
        lower_multiplier: float = 1
        upper_multiplier: float = 1

    if not (clean_series % 1 == 0).all():
        round_num: int = clean_series.astype(str).str.split(".", expand=True)[1].str.len().max()
    else:
        round_num: int = 0

    q1, q3 = clean_series.quantile([0.25, 0.75])
    iqr: float = q3 - q1
    outer_upper_fence: float = q3 + 3 * iqr * upper_multiplier
    inner_upper_fence: float = q3 + 1.5 * iqr * upper_multiplier
    inner_lower_fence: float = q1 - 1.5 * iqr * lower_multiplier
    outer_lower_fence: float = q1 - 3 * iqr * lower_multiplier

    return Fences(
        outer_lower=round(outer_lower_fence, round_num),
        inner_lower=round(inner_lower_fence, round_num),
        inner_upper=round(inner_upper_fence, round_num),
        outer_upper=round(outer_upper_fence, round_num)
    )
