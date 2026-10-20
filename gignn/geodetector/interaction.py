from __future__ import annotations

from itertools import combinations
import math
from typing import Iterable

import numpy as np
import pandas as pd

from gignn.exceptions import ValidationError
from gignn.geodetector.factor import _q_p_value, geodetector_q, target_to_numeric

_Q_TOLERANCE = 1e-12


def interaction_detector(
    df: pd.DataFrame, y: str, factors: Iterable[str]
) -> pd.DataFrame:
    if y not in df.columns:
        raise ValidationError(f"Target column {y!r} does not exist.")
    factor_names = list(factors)
    for factor in factor_names:
        if factor not in df.columns:
            raise ValidationError(f"Factor column {factor!r} does not exist.")
    y_values = target_to_numeric(df[y])
    rows = []
    for factor_1, factor_2 in combinations(factor_names, 2):
        values_1 = df[factor_1].to_numpy()
        values_2 = df[factor_2].to_numpy()
        combined = np.array(
            [f"{left}␟{right}" for left, right in zip(values_1, values_2)], dtype=object
        )
        q_1 = geodetector_q(y_values, values_1)
        q_2 = geodetector_q(y_values, values_2)
        q_interaction = geodetector_q(y_values, combined)
        rows.append(
            {
                "factor_1": factor_1,
                "factor_2": factor_2,
                "q_factor_1": q_1,
                "q_factor_2": q_2,
                "q_a": q_1,
                "q_b": q_2,
                "q_interaction": q_interaction,
                "p_value_interaction": _q_p_value(y_values, combined, q_interaction),
                "interaction": _classify_interaction(q_1, q_2, q_interaction),
            }
        )
    return pd.DataFrame(rows)


def _classify_interaction(q_1: float, q_2: float, q_interaction: float) -> str:
    # Standard GeoDetector interaction table: independent requires q12 == q1 + q2,
    # bi-enhance requires max(q1, q2) < q12 < q1 + q2. Because the combined
    # stratification refines each marginal one, q12 >= max(q1, q2) always holds,
    # so q12 == max means the weaker factor contributes nothing (weaken).
    total = q_1 + q_2
    maximum = max(q_1, q_2)
    if math.isclose(q_interaction, total, rel_tol=1e-9, abs_tol=_Q_TOLERANCE):
        return "independent"
    if q_interaction > total:
        return "nonlinear_enhance"
    if q_interaction > maximum + _Q_TOLERANCE:
        return "bi_enhance"
    return "weaken"
