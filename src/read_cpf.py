import pandas as pd
from datetime import datetime, timedelta


def mjd_to_datetime(mjd, sod):
    """
    Convert Modified Julian Date (MJD) and
    Seconds of Day (SOD) to a Python datetime.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    sod : float
        Seconds of day.

    Returns
    -------
    datetime
    """

    # MJD epoch = 1858-11-17 00:00:00 UTC
    mjd_epoch = datetime(1858, 11, 17)

    return mjd_epoch + timedelta(days=mjd, seconds=sod)


def read_cpf(filename):
    """
    Read an ILRS CPF file.

    Returns
    -------
    DataFrame

    Columns
    -------
    timestamp
    x
    y
    z

    Positions are returned in kilometers.
    """

    data = []

    with open(filename, "r") as f:

        for line in f:

            # Position records begin with '10'
            if not line.startswith("10"):
                continue

            fields = line.split()

            mjd = float(fields[2])
            sod = float(fields[3])

            timestamp = mjd_to_datetime(mjd, sod)

            x = float(fields[5]) / 1000.0
            y = float(fields[6]) / 1000.0
            z = float(fields[7]) / 1000.0

            data.append(
                {
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    df = pd.DataFrame(data)

    return df