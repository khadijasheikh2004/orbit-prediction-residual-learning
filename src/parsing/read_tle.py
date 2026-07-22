from datetime import datetime, timedelta

from sgp4.api import Satrec


def read_tle_history(filename, name=None):
    """Read a Space-Track-style TLE history file for a single satellite.

    Unlike ``read_tle`` (which expects 3-line name+line1+line2 groups
    for possibly many satellites), this reads the bare 2-line format
    returned by the Space-Track ``gp_history`` query used by
    ``data_downloader/download_tles.py``: many chronological
    (line1, line2) pairs for a single satellite, with no name line.

    Parameters
    ----------
    filename : str or Path
        Path to a 2-line TLE history file.
    name : str, optional
        Satellite display name to attach to each record (the file
        itself carries no name).

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, ``epoch`` (datetime), ``line1``,
        ``line2``, ``satellite`` (Satrec object), ordered by epoch.
    """

    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    lines = [line for line in lines if line]

    if len(lines) % 2 != 0:
        raise ValueError(
            "Invalid TLE history file. Every entry must have 2 lines."
        )

    records = []

    for i in range(0, len(lines), 2):

        line1 = lines[i]
        line2 = lines[i + 1]

        satellite = Satrec.twoline2rv(line1, line2)

        epoch_year = satellite.epochyr
        epoch_year += 2000 if epoch_year < 57 else 1900

        epoch = datetime(epoch_year, 1, 1) + timedelta(
            days=satellite.epochdays - 1
        )

        records.append(
            {
                "name": name,
                "epoch": epoch,
                "line1": line1,
                "line2": line2,
                "satellite": satellite,
            }
        )

    records.sort(key=lambda r: r["epoch"])

    return records


def read_tle(filename):
    """
    Reads a TLE file containing multiple satellites.

    Returns
    -------
    satellites : list
        List of dictionaries containing:
        - name
        - line1
        - line2
        - satellite (Satrec object)
    """

    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Remove blank lines
    lines = [line for line in lines if line]

    if len(lines) % 3 != 0:
        raise ValueError(
            "Invalid TLE file. Every satellite must have 3 lines."
        )

    satellites = []

    for i in range(0, len(lines), 3):

        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        satellite = Satrec.twoline2rv(line1, line2)

        satellites.append(
            {
                "name": name,
                "line1": line1,
                "line2": line2,
                "satellite": satellite,
            }
        )

    return satellites