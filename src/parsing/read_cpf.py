"""Raw ILRS Consolidated Prediction Format (CPF) parser.

This module has exactly one responsibility: read the position records
out of a CPF file and hand back raw, unmodified ITRF coordinates.

It intentionally performs NO interpolation and NO frame conversion.
Those responsibilities live in ``src/interpolation`` and
``src/orbit/coordinate_frames.py`` respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# MJD epoch: 1858-11-17 00:00:00 UTC
_MJD_EPOCH = datetime(1858, 11, 17)

# CPF position records begin with this record identifier.
_POSITION_RECORD_ID = "10"


@dataclass(frozen=True)
class CpfRecord:
    """A single raw CPF position record (ITRF, kilometers)."""

    timestamp: datetime
    x_itrf: float
    y_itrf: float
    z_itrf: float


def _mjd_sod_to_datetime(mjd: float, sod: float) -> datetime:
    """Convert Modified Julian Date + Seconds-of-Day to a UTC datetime."""

    return _MJD_EPOCH + timedelta(days=mjd, seconds=sod)


def read_cpf(filename: str | Path) -> pd.DataFrame:
    """Read raw position records from a single ILRS CPF file.

    Parameters
    ----------
    filename : str or Path
        Path to a CPF prediction file.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``x_itrf``, ``y_itrf``, ``z_itrf``.
        Positions are in kilometers, expressed in the ITRF (Earth-fixed)
        frame exactly as published in the CPF file. No interpolation or
        frame conversion is applied.
    """

    filename = Path(filename)

    records: list[CpfRecord] = []

    with filename.open("r") as f:
        for line in f:
            # Position records begin with record type '10'.
            if not line.startswith(_POSITION_RECORD_ID):
                continue

            fields = line.split()

            if len(fields) < 8:
                logger.warning(
                    "Skipping malformed CPF position record in %s: %r",
                    filename,
                    line,
                )
                continue

            mjd = float(fields[2])
            sod = float(fields[3])

            timestamp = _mjd_sod_to_datetime(mjd, sod)

            # Fields are in meters in the CPF file; convert to kilometers.
            x_itrf = float(fields[5]) / 1000.0
            y_itrf = float(fields[6]) / 1000.0
            z_itrf = float(fields[7]) / 1000.0

            records.append(
                CpfRecord(
                    timestamp=timestamp,
                    x_itrf=x_itrf,
                    y_itrf=y_itrf,
                    z_itrf=z_itrf,
                )
            )

    if not records:
        logger.warning("No CPF position records found in %s", filename)

    df = pd.DataFrame(
        {
            "timestamp": [r.timestamp for r in records],
            "x_itrf": [r.x_itrf for r in records],
            "y_itrf": [r.y_itrf for r in records],
            "z_itrf": [r.z_itrf for r in records],
        }
    )

    return df