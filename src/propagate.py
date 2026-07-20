from sgp4.api import jday


def propagate_satellite(satellite, timestamps):
    """
    Propagate a satellite at specified timestamps.

    Parameters
    ----------
    satellite : Satrec
        SGP4 satellite.

    timestamps : iterable of datetime
        Epochs at which to propagate.

    Returns
    -------
    list of dictionaries
    """

    predictions = []

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
            continue

        predictions.append(
            {
                "timestamp": current_time,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "vx": velocity[0],
                "vy": velocity[1],
                "vz": velocity[2],
            }
        )

    return predictions