"""SGP4 propagation.

Single responsibility: propagate a TLE-derived satellite state using
SGP4 and return the result strictly in its native TEME frame.

This module performs NO frame conversion. TEME -> GCRS conversion is
the responsibility of ``src/orbit/coordinate_frames.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemeState:
    """A single propagated SGP4 state, in TEME."""

    timestamp: datetime
    x_teme: float
    y_teme: float
    z_teme: float
    vx_teme: float
    vy_teme: float
    vz_teme: float


def propagate_satellite(
    satellite: Satrec,
    timestamps: Iterable[datetime],
) -> pd.DataFrame:
    """Propagate a satellite at specified timestamps using SGP4.

    Parameters
    ----------
    satellite : Satrec
        SGP4 satellite record (see ``parsing.read_tle.read_tle``).
    timestamps : iterable of datetime
        UTC epochs at which to propagate.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``x_teme``, ``y_teme``, ``z_teme``,
        ``vx_teme``, ``vy_teme``, ``vz_teme``.
        Position is in kilometers, velocity in kilometers/second, both
        strictly in the TEME frame native to SGP4. No frame conversion
        is applied here.
    """

    states: list[TemeState] = []

    for current_time in timestamps:

        jd, fr = jday(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second + current_time.microsecond / 1e6,
        )

        error, position, velocity = satellite.sgp4(jd, fr)

        if error != 0:
            logger.warning(
                "SGP4 error code %s at %s; skipping epoch.",
                error,
                current_time,
            )
            continue

        states.append(
            TemeState(
                timestamp=current_time,
                x_teme=position[0],
                y_teme=position[1],
                z_teme=position[2],
                vx_teme=velocity[0],
                vy_teme=velocity[1],
                vz_teme=velocity[2],
            )
        )

    return pd.DataFrame(
        {
            "timestamp": [s.timestamp for s in states],
            "x_teme": [s.x_teme for s in states],
            "y_teme": [s.y_teme for s in states],
            "z_teme": [s.z_teme for s in states],
            "vx_teme": [s.vx_teme for s in states],
            "vy_teme": [s.vy_teme for s in states],
            "vz_teme": [s.vz_teme for s in states],
        }
    )