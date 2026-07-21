import os
import requests

from dotenv import load_dotenv

from config import (
    SATELLITES,
    TLE_DIR,
    YEAR,
)


# ------------------------------------------------------------
# Load .env from the project root
# ------------------------------------------------------------

load_dotenv()


USERNAME = os.getenv(
    "SPACETRACK_USERNAME"
)

PASSWORD = os.getenv(
    "SPACETRACK_PASSWORD"
)


LOGIN_URL = (
    "https://www.space-track.org/ajaxauth/login"
)


def download_tle(
    session,
    satellite_key,
    satellite_info
):

    norad_id = satellite_info[
        "norad_id"
    ]

    output_dir = (
        TLE_DIR
        / satellite_key
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{YEAR}.tle"
    )

    print(
        f"\nDownloading TLEs for "
        f"{satellite_info['name']}"
    )

    print(
        f"NORAD ID: {norad_id}"
    )

    url = (

        "https://www.space-track.org/"
        "basicspacedata/query/"

        f"class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"

        f"EPOCH/%3E%3D"
        f"{YEAR}-01-01T00:00:00/"

        f"EPOCH/%3C"
        f"{YEAR + 1}-01-01T00:00:00/"

        "orderby/EPOCH asc/"
        "format/tle"
    )

    response = session.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    if not response.text.strip():

        print(
            f"No TLE data returned for "
            f"{satellite_info['name']}"
        )

        return

    output_file.write_text(
        response.text
    )

    print(
        f"Saved: {output_file}"
    )


def main():

    print(
        "Starting TLE downloader..."
    )

    # --------------------------------------------------------
    # Check credentials
    # --------------------------------------------------------

    if not USERNAME:

        raise RuntimeError(
            "SPACETRACK_USERNAME was not "
            "found in .env"
        )

    if not PASSWORD:

        raise RuntimeError(
            "SPACETRACK_PASSWORD was not "
            "found in .env"
        )

    print(
        f"Username loaded: {USERNAME}"
    )

    print(
        "Password loaded: YES"
    )

    # --------------------------------------------------------
    # Create session
    # --------------------------------------------------------

    session = requests.Session()

    login_data = {

        "identity": USERNAME,
        "password": PASSWORD,
    }

    print(
        "\nLogging into Space-Track..."
    )

    response = session.post(

        LOGIN_URL,

        data=login_data,

        timeout=60
    )

    response.raise_for_status()

    print(
        "Space-Track login successful."
    )

    # --------------------------------------------------------
    # Download satellite TLEs
    # --------------------------------------------------------

    for satellite_key, satellite_info in SATELLITES.items():

        try:

            download_tle(

                session,

                satellite_key,

                satellite_info
            )

        except Exception as error:

            print(
                f"\nERROR downloading "
                f"{satellite_info['name']}"
            )

            print(error)

            print(
                "Continuing to next satellite..."
            )


if __name__ == "__main__":

    main()