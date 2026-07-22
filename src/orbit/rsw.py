"""RSW (Radial / Along-track / Cross-track) frame transformation.

Single responsibility: build the RSW rotation basis from a predicted
(SGP4) state vector and rotate residual vectors into that frame, as
described in the paper. The RSW frame is defined, per the paper, as:

    R (radial)      : along the position vector, Earth center -> satellite
    S (along-track)  : in the orbital plane, perpendicular to R,
                       pointing (approximately) along the inertial
                       velocity direction
    W (cross-track) : along the orbital angular momentum vector
                       (r x v)

The RSW frame is built from the PREDICTED state, not the truth state,
matching the paper's definition of e_T expressed "in the RSW frame
centered at the predicted state".
"""

from __future__ import annotations

import numpy as np


def eci_to_rsw_rotation(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
) -> np.ndarray:
    """Build the rotation matrix from ECI (e.g. GCRS) to RSW.

    Parameters
    ----------
    position_eci : array-like, shape (3,)
        Position vector in an Earth-centered inertial frame.
    velocity_eci : array-like, shape (3,)
        Velocity vector in the same inertial frame.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix ``M`` such that, for any vector ``v`` expressed
        in the inertial frame, ``v_rsw = M @ v_eci`` gives that vector
        expressed in the RSW frame. Rows of ``M`` are the RSW unit
        basis vectors (R, S, W), expressed in the inertial frame.
    """

    r = np.asarray(position_eci, dtype=float)
    v = np.asarray(velocity_eci, dtype=float)

    r_hat = r / np.linalg.norm(r)

    w = np.cross(r, v)
    w_hat = w / np.linalg.norm(w)

    s_hat = np.cross(w_hat, r_hat)
    s_hat = s_hat / np.linalg.norm(s_hat)

    rotation = np.vstack([r_hat, s_hat, w_hat])

    return rotation


def residual_to_rsw(
    residual_position: np.ndarray,
    residual_velocity: np.ndarray,
    predicted_position: np.ndarray,
    predicted_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a position/velocity residual vector into the RSW frame.

    The RSW basis is built from the PREDICTED state (per the paper's
    definition of e_T), and the same fixed rotation is applied to both
    the position and velocity residual vectors.

    Parameters
    ----------
    residual_position : array-like, shape (3,)
        ``(dx, dy, dz)`` residual (Truth - Prediction), inertial frame.
    residual_velocity : array-like, shape (3,)
        ``(dvx, dvy, dvz)`` residual (Truth - Prediction), inertial frame.
    predicted_position : array-like, shape (3,)
        Predicted (SGP4) position, inertial frame, used to build the
        RSW basis.
    predicted_velocity : array-like, shape (3,)
        Predicted (SGP4) velocity, inertial frame, used to build the
        RSW basis.

    Returns
    -------
    (residual_position_rsw, residual_velocity_rsw) : tuple of np.ndarray
        Each of shape (3,): ``(e_r, e_s, e_w)`` for position and
        velocity residuals respectively.
    """

    rotation = eci_to_rsw_rotation(predicted_position, predicted_velocity)

    residual_position_rsw = rotation @ np.asarray(residual_position, dtype=float)
    residual_velocity_rsw = rotation @ np.asarray(residual_velocity, dtype=float)

    return residual_position_rsw, residual_velocity_rsw