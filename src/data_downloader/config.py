from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

YEAR = 2025


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


TLE_DIR = (
    DATA_DIR
    / "tle"
)


CPF_DIR = (
    DATA_DIR
    / "cpf"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

TLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CPF_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SATELLITES
# ============================================================

SATELLITES = {

    "ajisai": {

        "name": "AJISAI",

        "norad_id": 16908,

        "ilrs_id": "8606101",

        "provider": "SGF",
    },

    "beaconc": {

        "name":
        "EXPLORER 27 (BEACON-C)",

        "norad_id": 1328,

        "ilrs_id": "6503201",

        "provider": "HTS",
    },

    "etalon1": {

        "name":
        "COSMOS 1989 (ETALON 1)",

        "norad_id": 19751,

        "ilrs_id": "8900103",

        "provider": "DGF",
    },

    "etalon2": {

        "name":
        "COSMOS 2024 (ETALON 2)",

        "norad_id": 20026,

        "ilrs_id": "8903903",

        "provider": "DGF",
    },

    "lageos1": {

        "name":
        "LAGEOS 1",

        "norad_id": 8820,

        "ilrs_id": "7603901",

        "provider": "HTS",
    },

    "lageos2": {

        "name":
        "LAGEOS 2",

        "norad_id": 22195,

        "ilrs_id": "9207001",

        "provider": "DGF",
    },

    "lares": {

        "name":
        "LARES",

        "norad_id": 38077,

        "ilrs_id": "1200601",

        "provider": "DGF",
    },

    "larets": {

        "name":
        "LARETS",

        "norad_id": 27944,

        "ilrs_id": "0304206",

        "provider": "SGF",
    },

    "starlette": {

        "name":
        "STARLETTE",

        "norad_id": 7646,

        "ilrs_id": "7501001",

        "provider": "DGF",
    },

    "stella": {

        "name":
        "STELLA",

        "norad_id": 22824,

        "ilrs_id": "9306102",

        "provider": "DGF",
    },
}