"""Coordinate frame conversions for orbit states.

Single responsibility: transform position/velocity vectors between
reference frames. This module performs NO parsing, NO interpolation,
and NO propagation.

All Earth-orientation and precession/nutation modeling is delegated to
Astropy (``astropy.coordinates``); no manual Earth-orientation
implementation is done here.

Frames used in this project:

* TEME  - True Equator, Mean Equinox (native SGP4 output frame).
* ITRF  - International Terrestrial Reference Frame (native CPF frame).
* GCRS  - Geocentric Celestial Reference System, used as the common
          inertial frame in which residuals are computed.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, TEME, CartesianDifferential, CartesianRepresentation
from astropy.time import Time


def _make_time(epoch: datetime) -> Time:
    return Time(epoch, scale="utc")


def _make_cartesian(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
) -> CartesianRepresentation:
    return CartesianRepresentation(
        x=position_km[0] * u.km,
        y=position_km[1] * u.km,
        z=position_km[2] * u.km,
        differentials=CartesianDifferential(
            d_x=velocity_km_s[0] * u.km / u.s,
            d_y=velocity_km_s[1] * u.km / u.s,
            d_z=velocity_km_s[2] * u.km / u.s,
        ),
    )


def _extract(
    cartesian: CartesianRepresentation,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [
            cartesian.x.to_value(u.km),
            cartesian.y.to_value(u.km),
            cartesian.z.to_value(u.km),
        ]
    )

    diff = cartesian.differentials["s"]

    velocity = np.array(
        [
            diff.d_x.to_value(u.km / u.s),
            diff.d_y.to_value(u.km / u.s),
            diff.d_z.to_value(u.km / u.s),
        ]
    )

    return position, velocity


def teme_to_gcrs(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    epoch: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a TEME state vector to GCRS.

    Parameters
    ----------
    position_km : array-like, shape (3,)
        Position in TEME, kilometers.
    velocity_km_s : array-like, shape (3,)
        Velocity in TEME, kilometers/second.
    epoch : datetime
        UTC epoch of the state vector.

    Returns
    -------
    (position_gcrs_km, velocity_gcrs_km_s) : tuple of np.ndarray
    """

    time = _make_time(epoch)
    cart = _make_cartesian(np.asarray(position_km), np.asarray(velocity_km_s))

    teme = TEME(cart, obstime=time)
    gcrs = teme.transform_to(GCRS(obstime=time))

    return _extract(gcrs.cartesian)


def gcrs_to_teme(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    epoch: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a GCRS state vector to TEME.

    Parameters
    ----------
    position_km : array-like, shape (3,)
        Position in GCRS, kilometers.
    velocity_km_s : array-like, shape (3,)
        Velocity in GCRS, kilometers/second.
    epoch : datetime
        UTC epoch of the state vector.

    Returns
    -------
    (position_teme_km, velocity_teme_km_s) : tuple of np.ndarray
    """

    time = _make_time(epoch)
    cart = _make_cartesian(np.asarray(position_km), np.asarray(velocity_km_s))

    gcrs = GCRS(cart, obstime=time)
    teme = gcrs.transform_to(TEME(obstime=time))

    return _extract(teme.cartesian)


def itrf_to_gcrs(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    epoch: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ITRF (Earth-fixed) state vector to GCRS.

    Parameters
    ----------
    position_km : array-like, shape (3,)
        Position in ITRF, kilometers.
    velocity_km_s : array-like, shape (3,)
        Velocity in ITRF, kilometers/second.
    epoch : datetime
        UTC epoch of the state vector.

    Returns
    -------
    (position_gcrs_km, velocity_gcrs_km_s) : tuple of np.ndarray
    """

    time = _make_time(epoch)
    cart = _make_cartesian(np.asarray(position_km), np.asarray(velocity_km_s))

    itrs = ITRS(cart, obstime=time)
    gcrs = itrs.transform_to(GCRS(obstime=time))

    return _extract(gcrs.cartesian)


def gcrs_to_itrf(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    epoch: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a GCRS state vector to ITRF (Earth-fixed).

    Parameters
    ----------
    position_km : array-like, shape (3,)
        Position in GCRS, kilometers.
    velocity_km_s : array-like, shape (3,)
        Velocity in GCRS, kilometers/second.
    epoch : datetime
        UTC epoch of the state vector.

    Returns
    -------
    (position_itrf_km, velocity_itrf_km_s) : tuple of np.ndarray
    """

    time = _make_time(epoch)
    cart = _make_cartesian(np.asarray(position_km), np.asarray(velocity_km_s))

    gcrs = GCRS(cart, obstime=time)
    itrs = gcrs.transform_to(ITRS(obstime=time))

    return _extract(itrs.cartesian)