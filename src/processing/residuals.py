"""Compute position/velocity residuals.

Single responsibility: given epoch-matched prediction and truth
states, compute

    Residual = Truth - Prediction

for each of the six state components. No frame conversion, matching,
or interpolation is performed here.
"""

from __future__ import annotations

import pandas as pd

_COMPONENTS = ["x", "y", "z", "vx", "vy", "vz"]
_RESIDUAL_NAMES = ["dx", "dy", "dz", "dvx", "dvy", "dvz"]


def compute_residuals(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Truth - Prediction residuals for matched states.

    Parameters
    ----------
    matched_df : pandas.DataFrame
        Output of ``processing.epoch_match.epoch_match``: must contain
        ``timestamp`` plus ``{x,y,z,vx,vy,vz}_pred`` and
        ``{x,y,z,vx,vy,vz}_truth`` columns, all in a common frame
        (GCRS).

    Returns
    -------
    pandas.DataFrame
        ``timestamp`` plus the original matched columns, plus the
        residual columns ``dx``, ``dy``, ``dz``, ``dvx``, ``dvy``,
        ``dvz`` (Truth - Prediction; km and km/s).
    """

    result = matched_df.copy()

    for component, residual_name in zip(_COMPONENTS, _RESIDUAL_NAMES):
        result[residual_name] = (
            matched_df[f"{component}_truth"] - matched_df[f"{component}_pred"]
        )

    return result