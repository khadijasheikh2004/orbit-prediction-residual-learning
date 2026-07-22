"""Build "prediction" state vectors from TLE data.

Pipeline (single responsibility: combine, not compute the individual
steps):

    TLE
        -> SGP4 propagation (native TEME)
        -> TEME -> GCRS frame conversion
        -> save

This module does not implement propagation or frame-conversion math
itself; it only orchestrates ``src/propagation`` and ``src/orbit``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sgp4.api import Satrec

from orbit.coordinate_frames import teme_to_gcrs
from orbit.propagate import propagate_satellite

logger = logging.getLogger(__name__)


def build_prediction_states(
    satellite: Satrec,
    epochs: Iterable[datetime],
) -> pd.DataFrame:
    """Build GCRS prediction states by propagating a TLE with SGP4.

    Parameters
    ----------
    satellite : Satrec
        SGP4 satellite record (see ``parsing.read_tle.read_tle``).
    epochs : iterable of datetime
        Epochs at which to propagate.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``x``, ``y``, ``z``, ``vx``, ``vy``,
        ``vz``. Position in kilometers, velocity in kilometers/second,
        expressed in the GCRS frame.
    """

    teme_df = propagate_satellite(satellite, epochs)

    if teme_df.empty:
        logger.warning("SGP4 propagation produced no valid states.")
        return pd.DataFrame(
            columns=["timestamp", "x", "y", "z", "vx", "vy", "vz"]
        )

    rows = []

    for record in teme_df.itertuples(index=False):
        position_teme = np.array([record.x_teme, record.y_teme, record.z_teme])
        velocity_teme = np.array(
            [record.vx_teme, record.vy_teme, record.vz_teme]
        )

        position_gcrs, velocity_gcrs = teme_to_gcrs(
            position_teme, velocity_teme, record.timestamp
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