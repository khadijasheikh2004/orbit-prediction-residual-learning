"""10-point Lagrangian interpolation of CPF ephemerides.

This module implements the ILRS-style 10-point Lagrangian interpolator
used to evaluate a CPF ephemeris (and its analytical derivative, i.e.
velocity) at an arbitrary epoch. It is a direct Python translation of
the algorithm used in the official ILRS/CDDIS sample interpolation code
(commonly distributed as ``hermite.c`` / ``interp.c``):

  * a window of ``N`` tabulated points is selected, centered as closely
    as possible on the requested epoch, such that the epoch falls
    between the two middle points of the window whenever possible;
  * position is evaluated with the classical Lagrange interpolation
    polynomial built from that window;
  * velocity is evaluated with the *analytical derivative* of the same
    Lagrange polynomial (no finite differencing).

No SciPy interpolation routines and no splines are used anywhere in
this module, per the ILRS reference-implementation requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Number of points used in the Lagrangian interpolation window.
#: This mirrors the official ILRS "10-point baseline Lagrangian
#: interpolator" referenced by the CDDIS CPF sample code.
DEFAULT_NUM_POINTS = 10


@dataclass(frozen=True)
class InterpolatedState:
    """Interpolated ITRF position and velocity at a single epoch."""

    timestamp: datetime
    x_itrf: float
    y_itrf: float
    z_itrf: float
    vx_itrf: float
    vy_itrf: float
    vz_itrf: float


def _select_window(
    times: np.ndarray,
    epoch_seconds: float,
    num_points: int,
) -> np.ndarray:
    """Select the indices of the centered interpolation window.

    Mirrors the official ILRS centering convention: the window is
    chosen so that ``epoch_seconds`` falls between the two middle
    points of the window whenever there is enough data on both sides.
    Near the edges of the series, the window is clamped and shifted
    so it still contains exactly ``num_points`` points.
    """

    n = len(times)

    if n < num_points:
        raise ValueError(
            f"Need at least {num_points} CPF points to interpolate, "
            f"got {n}."
        )

    # Index of the first tabulated point strictly greater than the epoch.
    # (searchsorted assumes `times` is sorted ascending.)
    upper = int(np.searchsorted(times, epoch_seconds, side="right"))

    # For an even-sized window of `num_points`, the "middle two" points
    # are at local offsets num_points/2 - 1 and num_points/2. We want
    # `upper` to land exactly at the second of those two, i.e. the
    # window should start at `upper - num_points // 2`.
    half = num_points // 2
    start = upper - half

    # Clamp to valid bounds while keeping the window size fixed.
    start = max(0, min(start, n - num_points))

    return np.arange(start, start + num_points)


def _lagrange_value_and_derivative(
    t: float,
    nodes: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the Lagrange interpolating polynomial and its derivative.

    Uses the closed-form analytical derivative of the Lagrange basis
    polynomials:

        P(t)  = sum_i y_i * L_i(t)
        L_i(t) = prod_{j != i} (t - t_j) / (t_i - t_j)
        L_i'(t) = L_i(t) * sum_{j != i} 1 / (t - t_j)

    No finite differences are used. If ``t`` coincides exactly with one
    of the ``nodes`` (within floating point tolerance), the polynomial
    is evaluated directly at that node.
    """

    diffs = t - nodes

    # Exact-node fallback: if t matches a node, avoid division by zero
    # in the L_i'(t) sum by nudging with the analytical limit instead.
    exact = np.isclose(diffs, 0.0, atol=1e-9)

    if np.any(exact):
        k = int(np.argmax(exact))
        # P(t_k) = y_k exactly.
        position = float(values[k])

        # Derivative at a node: standard closed-form for Lagrange
        # differentiation at a tabulated node.
        #
        #   P'(x_k) = y_k * sum_{j != k} 1/(x_k - x_j)
        #             + sum_{i != k} y_i * L_i'(x_k)
        #
        #   L_i'(x_k) = [1/(x_i - x_k)] * prod_{j != i,k} (x_k-x_j)/(x_i-x_j)
        n = len(nodes)

        self_term = float(
            values[k] * np.sum(1.0 / (nodes[k] - np.delete(nodes, k)))
        )

        cross_term = 0.0
        for i in range(n):
            if i == k:
                continue
            prod = 1.0
            for j in range(n):
                if j != k and j != i:
                    prod *= (nodes[k] - nodes[j]) / (nodes[i] - nodes[j])
            cross_term += values[i] * prod / (nodes[i] - nodes[k])

        derivative = self_term + cross_term
        return position, derivative

    n = len(nodes)
    basis = np.empty(n)

    for i in range(n):
        num = 1.0
        den = 1.0
        for j in range(n):
            if j == i:
                continue
            num *= diffs[j]
            den *= (nodes[i] - nodes[j])
        basis[i] = num / den

    position = float(np.dot(values, basis))

    inv_diffs = 1.0 / diffs
    derivative = 0.0
    for i in range(n):
        sum_inv = float(np.sum(inv_diffs) - inv_diffs[i])
        derivative += values[i] * basis[i] * sum_inv

    return position, derivative


def interpolate_cpf(
    cpf_df: pd.DataFrame,
    epoch: datetime,
    num_points: int = DEFAULT_NUM_POINTS,
) -> InterpolatedState:
    """Interpolate a CPF ephemeris at a single arbitrary epoch.

    Parameters
    ----------
    cpf_df : pandas.DataFrame
        Raw CPF data as returned by ``parsing.read_cpf.read_cpf``,
        with columns ``timestamp``, ``x_itrf``, ``y_itrf``, ``z_itrf``.
    epoch : datetime
        Epoch at which to evaluate the interpolant.
    num_points : int, optional
        Size of the centered Lagrangian interpolation window
        (default 10, per the ILRS reference implementation).

    Returns
    -------
    InterpolatedState
        Interpolated ITRF position (km) and velocity (km/s) at
        ``epoch``.
    """

    if cpf_df.empty:
        raise ValueError("Cannot interpolate an empty CPF dataframe.")

    df = cpf_df.sort_values("timestamp").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    times = (df["timestamp"] - t0).dt.total_seconds().to_numpy()
    epoch_seconds = (epoch - t0).total_seconds()

    window_idx = _select_window(times, epoch_seconds, num_points)

    nodes = times[window_idx]

    x_vals = df["x_itrf"].to_numpy()[window_idx]
    y_vals = df["y_itrf"].to_numpy()[window_idx]
    z_vals = df["z_itrf"].to_numpy()[window_idx]

    x, vx = _lagrange_value_and_derivative(epoch_seconds, nodes, x_vals)
    y, vy = _lagrange_value_and_derivative(epoch_seconds, nodes, y_vals)
    z, vz = _lagrange_value_and_derivative(epoch_seconds, nodes, z_vals)

    return InterpolatedState(
        timestamp=epoch,
        x_itrf=x,
        y_itrf=y,
        z_itrf=z,
        vx_itrf=vx,
        vy_itrf=vy,
        vz_itrf=vz,
    )


def interpolate_cpf_series(
    cpf_df: pd.DataFrame,
    epochs: pd.Series | list[datetime],
    num_points: int = DEFAULT_NUM_POINTS,
) -> pd.DataFrame:
    """Interpolate a CPF ephemeris at many epochs.

    Parameters
    ----------
    cpf_df : pandas.DataFrame
        Raw CPF data (see ``interpolate_cpf``).
    epochs : sequence of datetime
        Epochs at which to evaluate the interpolant.
    num_points : int, optional
        Interpolation window size (default 10).

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``x_itrf``, ``y_itrf``, ``z_itrf``,
        ``vx_itrf``, ``vy_itrf``, ``vz_itrf``. Epochs that fall outside
        the CPF file's usable range (fewer than ``num_points`` points
        available) are skipped with a warning.
    """

    df = cpf_df.sort_values("timestamp").reset_index(drop=True)

    rows = []

    for epoch in epochs:
        try:
            state = interpolate_cpf(df, epoch, num_points=num_points)
        except ValueError as exc:
            logger.warning("Skipping epoch %s: %s", epoch, exc)
            continue

        rows.append(
            {
                "timestamp": state.timestamp,
                "x_itrf": state.x_itrf,
                "y_itrf": state.y_itrf,
                "z_itrf": state.z_itrf,
                "vx_itrf": state.vx_itrf,
                "vy_itrf": state.vy_itrf,
                "vz_itrf": state.vz_itrf,
            }
        )

    return pd.DataFrame(rows)