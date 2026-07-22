"""End-to-end residual-learning preprocessing pipeline.

    Read CPF
        -> Read TLE
        -> Interpolate CPF (10-point Lagrangian, ITRF)
        -> Convert CPF (truth) ITRF -> GCRS
        -> Propagate TLE with SGP4 (TEME)
        -> Convert TEME -> GCRS
        -> Merge epochs (prediction vs truth)
        -> Compute residuals (Truth - Prediction)
        -> Transform residuals to RSW
        -> Save
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "data_downloader"))

from data_downloader import config  # noqa: E402  (data_downloader/config.py: SATELLITES, CPF_DIR, TLE_DIR)
from analysis.plot_orbit import plot_orbit  # noqa: E402
from orbit.rsw import residual_to_rsw  # noqa: E402
from parsing.read_cpf import read_cpf  # noqa: E402
from parsing.read_tle import read_tle_history  # noqa: E402
from processing.build_prediction_states import build_prediction_states  # noqa: E402
from processing.build_truth_states import build_truth_states  # noqa: E402
from processing.epoch_match import epoch_match  # noqa: E402
from processing.residuals import compute_residuals  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for one pipeline execution."""

    output_dir: Path = SRC_DIR.parent / "output"
    prediction_horizon: timedelta = timedelta(hours=6)
    prediction_step: timedelta = timedelta(minutes=5)
    satellite_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(config.SATELLITES.keys())
    )


def _cpf_files_covering_window(
    satellite_key: str,
    window_start: datetime,
    window_end: datetime,
    margin: timedelta = timedelta(days=1),
) -> list[Path]:
    """Select the daily CPF files whose date could cover the window.

    CPF filenames embed the file's date as ``..._cpf_YYMMDD_*``. Rather
    than parsing every file for a satellite (there can be 500-900 daily
    files per year), only the handful of files whose date falls within
    ``[window_start - margin, window_end + margin]`` are selected. A
    margin is kept because a CPF file's own coverage can extend beyond
    its filename date (see Fig. 2 of the paper -- overlapping CPF
    validity windows).
    """

    sat_dir = config.CPF_DIR / satellite_key

    lower = (window_start - margin).date()
    upper = (window_end + margin).date()

    selected = []

    for f in sorted(sat_dir.glob("*")):
        parts = f.stem.split("_")
        # Expected stem shape: "<sat>_cpf_YYMMDD_#####"
        date_token = next(
            (p for p in parts if len(p) == 6 and p.isdigit()), None
        )
        if date_token is None:
            continue

        file_date = datetime.strptime(date_token, "%y%m%d").date()

        if lower <= file_date <= upper:
            selected.append(f)

    return selected


def _load_cpf_states_for_window(
    satellite_key: str,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    """Load and concatenate the CPF daily files covering a time window.

    Duplicate timestamps (which occur where consecutive daily CPF
    files overlap) are resolved by keeping the most recently
    downloaded file's value, approximating the "most recently updated
    CPF file" rule described in the paper.
    """

    cpf_files = _cpf_files_covering_window(
        satellite_key, window_start, window_end
    )

    if not cpf_files:
        logger.warning(
            "No CPF files found for %s covering %s..%s",
            satellite_key,
            window_start,
            window_end,
        )
        return pd.DataFrame(columns=["timestamp", "x_itrf", "y_itrf", "z_itrf"])

    frames = [read_cpf(f) for f in cpf_files]
    combined = pd.concat(frames, ignore_index=True)

    combined = (
        combined.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return combined


def _prediction_epochs(
    start: datetime,
    horizon: timedelta,
    step: timedelta,
) -> list[datetime]:
    n_steps = int(horizon / step)
    return [start + i * step for i in range(n_steps + 1)]


def _add_rsw_columns(residual_df: pd.DataFrame) -> pd.DataFrame:
    """Rotate position/velocity residuals into the RSW frame (per row)."""

    df = residual_df.copy()

    rsw_position = np.zeros((len(df), 3))
    rsw_velocity = np.zeros((len(df), 3))

    for i, row in enumerate(df.itertuples(index=False)):
        residual_position = np.array([row.dx, row.dy, row.dz])
        residual_velocity = np.array([row.dvx, row.dvy, row.dvz])
        predicted_position = np.array([row.x_pred, row.y_pred, row.z_pred])
        predicted_velocity = np.array([row.vx_pred, row.vy_pred, row.vz_pred])

        pos_rsw, vel_rsw = residual_to_rsw(
            residual_position,
            residual_velocity,
            predicted_position,
            predicted_velocity,
        )

        rsw_position[i] = pos_rsw
        rsw_velocity[i] = vel_rsw

    df["dr"] = rsw_position[:, 0]
    df["ds"] = rsw_position[:, 1]
    df["dw"] = rsw_position[:, 2]
    df["dvr"] = rsw_velocity[:, 0]
    df["dvs"] = rsw_velocity[:, 1]
    df["dvw"] = rsw_velocity[:, 2]

    return df


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def process_satellite(
    satellite_key: str,
    satellite_info: dict,
    pipeline_config: PipelineConfig,
) -> pd.DataFrame | None:
    """Run the full residual-learning preprocessing pipeline for one RSO."""

    name = satellite_info["name"]

    logger.info("=" * 60)
    logger.info("%s (%s)", name, satellite_key)
    logger.info("=" * 60)

    tle_path = config.TLE_DIR / satellite_key / f"{config.YEAR}.tle"

    if not tle_path.exists():
        logger.warning("Missing TLE file: %s", tle_path)
        return None

    tle_records = read_tle_history(tle_path, name=name)

    if not tle_records:
        logger.warning("No TLE records found for %s", name)
        return None

    # Representative propagation window: use the earliest TLE epoch
    # that falls within the configured data year (TLE history files
    # can span back to the satellite's launch), propagated forward for
    # a fixed horizon. (Building the full multi-epoch training dataset
    # is a later stage -- see module docstring.)
    year_records = [r for r in tle_records if r["epoch"].year == config.YEAR]

    if not year_records:
        logger.warning(
            "No TLE records found for %s in year %d", name, config.YEAR
        )
        return None

    tle_record = year_records[0]
    satellite = tle_record["satellite"]

    epochs = _prediction_epochs(
        tle_record["epoch"],
        pipeline_config.prediction_horizon,
        pipeline_config.prediction_step,
    )

    cpf_df = _load_cpf_states_for_window(
        satellite_key, epochs[0], epochs[-1]
    )

    if cpf_df.empty:
        logger.warning("No CPF data available for %s", name)
        return None

    prediction_df = build_prediction_states(satellite, epochs)

    if prediction_df.empty:
        logger.warning("SGP4 produced no valid predictions for %s", name)
        return None

    truth_df = build_truth_states(cpf_df, prediction_df["timestamp"])

    if truth_df.empty:
        logger.warning("No truth states could be interpolated for %s", name)
        return None

    matched_df = epoch_match(prediction_df, truth_df)

    if matched_df.empty:
        logger.warning("No matched epochs between prediction and truth for %s", name)
        return None

    residual_df = compute_residuals(matched_df)
    residual_df = _add_rsw_columns(residual_df)
    residual_df.insert(0, "satellite", name)

    safe_name = _safe_name(name)

    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)

    residual_df.to_csv(
        pipeline_config.output_dir / f"{safe_name}_residuals.csv",
        index=False,
    )

    plot_orbit(
        prediction_df,
        name,
        str(pipeline_config.output_dir / f"{safe_name}_orbit.png"),
        truth_df,
    )

    logger.info(
        "%s: %d matched epochs, mean |dr|=%.4f km, mean |dvr|=%.6f km/s",
        name,
        len(residual_df),
        residual_df["dr"].abs().mean(),
        residual_df["dvr"].abs().mean(),
    )

    return residual_df


def main() -> None:
    pipeline_config = PipelineConfig()

    logger.info("=" * 60)
    logger.info("Residual Learning Preprocessing Pipeline")
    logger.info("=" * 60)

    all_residuals: list[pd.DataFrame] = []

    for satellite_key in pipeline_config.satellite_keys:
        satellite_info = config.SATELLITES[satellite_key]

        try:
            residual_df = process_satellite(
                satellite_key, satellite_info, pipeline_config
            )
        except Exception:
            logger.exception("Pipeline failed for %s", satellite_info["name"])
            continue

        if residual_df is not None:
            all_residuals.append(residual_df)

    if all_residuals:
        combined = pd.concat(all_residuals, ignore_index=True)
        combined.to_csv(
            pipeline_config.output_dir / "all_residuals_rsw.csv",
            index=False,
        )
        logger.info("Saved %d combined residual records.", len(combined))

    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()