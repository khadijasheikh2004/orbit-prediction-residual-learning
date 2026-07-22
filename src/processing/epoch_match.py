"""Align prediction and truth state vectors by timestamp.

Single responsibility: merge two already-computed state DataFrames on
their shared ``timestamp`` column. No interpolation is performed here
-- both inputs are expected to already be evaluated at the same set of
epochs (typically the SGP4 prediction epochs, with truth states built
at those exact epochs via ``build_truth_states``).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_STATE_COLUMNS = ["x", "y", "z", "vx", "vy", "vz"]


def epoch_match(
    prediction_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge prediction and truth states on exact timestamp match.

    Parameters
    ----------
    prediction_df : pandas.DataFrame
        Columns ``timestamp``, ``x``, ``y``, ``z``, ``vx``, ``vy``,
        ``vz`` (GCRS), as returned by ``build_prediction_states``.
    truth_df : pandas.DataFrame
        Columns ``timestamp``, ``x``, ``y``, ``z``, ``vx``, ``vy``,
        ``vz`` (GCRS), as returned by ``build_truth_states``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, then the prediction six-component
        state suffixed ``_pred``, then the truth six-component state
        suffixed ``_truth``. Only epochs present in both inputs are
        kept.
    """

    merged = pd.merge(
        prediction_df,
        truth_df,
        on="timestamp",
        suffixes=("_pred", "_truth"),
        how="inner",
    )

    if merged.empty:
        logger.warning(
            "epoch_match produced no aligned epochs "
            "(prediction=%d rows, truth=%d rows).",
            len(prediction_df),
            len(truth_df),
        )

    ordered_columns = ["timestamp"]
    ordered_columns += [f"{c}_pred" for c in _STATE_COLUMNS]
    ordered_columns += [f"{c}_truth" for c in _STATE_COLUMNS]

    return merged[ordered_columns].reset_index(drop=True)