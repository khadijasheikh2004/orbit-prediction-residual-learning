"""Build "truth" state vectors from CPF data.

Pipeline (single responsibility: combine, not compute the individual
steps):

    CPF (raw ITRF)
        -> 10-point Lagrangian interpolation (ITRF position + velocity)
        -> ITRF -> GCRS frame conversion
        -> save

This module does not implement interpolation or frame-conversion math
itself; it only orchestrates the modules in ``src/interpolation`` and
``src/orbit`` that do.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from interpolation.lagrange10 import interpolate_cpf_series
from orbit.coordinate_frames import itrf_to_gcrs

logger = logging.getLogger(__name__)


def build_truth_states(
    cpf_df: pd.DataFrame,
    epochs: Iterable[datetime],
) -> pd.DataFrame:
    """Build GCRS truth states from raw CPF data at the given epochs.

    Parameters
    ----------
    cpf_df : pandas.DataFrame
        Raw CPF data (``timestamp``, ``x_itrf``, ``y_itrf``, ``z_itrf``)
        as returned by ``parsing.read_cpf.read_cpf``.
    epochs : iterable of datetime
        Epochs at which to evaluate the truth ephemeris (typically the
        SGP4 prediction epochs).

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``x``, ``y``, ``z``, ``vx``, ``vy``,
        ``vz``. Position in kilometers, velocity in kilometers/second,
        expressed in the GCRS frame.
    """

    interpolated = interpolate_cpf_series(cpf_df, epochs)

    if interpolated.empty:
        logger.warning("No epochs could be interpolated from CPF data.")
        return pd.DataFrame(
            columns=["timestamp", "x", "y", "z", "vx", "vy", "vz"]
        )

    rows = []

    for record in interpolated.itertuples(index=False):
        position_itrf = np.array([record.x_itrf, record.y_itrf, record.z_itrf])
        velocity_itrf = np.array(
            [record.vx_itrf, record.vy_itrf, record.vz_itrf]
        )

        position_gcrs, velocity_gcrs = itrf_to_gcrs(
            position_itrf, velocity_itrf, record.timestamp
        )

        rows.append(
            {
                "timestamp": record.timestamp,
                "x": position_gcrs[0],
                "y": position_gcrs[1],
                "z": position_gcrs[2],
                "vx": velocity_gcrs[0],
                "vy": velocity_gcrs[1],
                "vz": velocity_gcrs[2],
            }
        )

    return pd.DataFrame(rows)