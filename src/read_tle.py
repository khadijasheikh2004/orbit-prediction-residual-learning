from sgp4.api import Satrec


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