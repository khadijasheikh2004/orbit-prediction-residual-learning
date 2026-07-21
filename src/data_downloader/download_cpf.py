import time
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import (
    SATELLITES,
    CPF_DIR,
    YEAR,
)


BASE_URL = (
    "https://edc.dgfi.tum.de/"
    "pub/slr/cpf_predicts_v2/"
    f"{YEAR}/"
)


def get_file_links(
    satellite_key,
    provider
):

    satellite_url = (
        BASE_URL
        + satellite_key
        + "/"
    )

    response = requests.get(
        satellite_url,
        timeout=60
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    links = []

    for link in soup.find_all("a"):

        href = link.get("href")

        if not href:
            continue

        filename = href.split("/")[-1]

        if not filename:
            continue

        if filename.lower().endswith(
            f".{provider.lower()}"
        ):

            links.append(
                urljoin(
                    satellite_url,
                    href
                )
            )

    return links


def download_file(
    url,
    output_file,
    max_retries=5
):

    # --------------------------------------
    # Skip already downloaded files
    # --------------------------------------

    if output_file.exists():

        print(
            f"Already exists: "
            f"{output_file.name}"
        )

        return

    # --------------------------------------
    # Try downloading multiple times
    # --------------------------------------

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            print(
                f"Downloading "
                f"{output_file.name} "
                f"(attempt "
                f"{attempt}/{max_retries})"
            )

            response = requests.get(
                url,
                timeout=120
            )

            response.raise_for_status()

            output_file.write_bytes(
                response.content
            )

            print(
                f"Downloaded: "
                f"{output_file.name}"
            )

            return

        except requests.RequestException as error:

            print(
                f"Attempt {attempt} failed "
                f"for {output_file.name}:"
            )

            print(error)

            # ------------------------------
            # Retry if attempts remain
            # ------------------------------

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            else:

                print(
                    f"Giving up on "
                    f"{output_file.name}"
                )


def process_satellite(
    satellite_key,
    satellite_info
):

    provider = satellite_info[
        "provider"
    ].lower()

    print(
        f"\n{satellite_info['name']}"
    )

    print(
        f"Provider: "
        f"{provider.upper()}"
    )

    # --------------------------------------
    # Get all CPF links
    # --------------------------------------

    links = get_file_links(
        satellite_key,
        provider
    )

    # --------------------------------------
    # Create satellite directory
    # --------------------------------------

    output_dir = (
        CPF_DIR
        / satellite_key
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Found {len(links)} files"
    )

    # --------------------------------------
    # Download each CPF file
    # --------------------------------------

    for url in links:

        filename = url.split("/")[-1]

        output_file = (
            output_dir
            / filename
        )

        download_file(
            url,
            output_file
        )


def main():

    for satellite_key, satellite_info in SATELLITES.items():

        try:

            process_satellite(
                satellite_key,
                satellite_info
            )

        except requests.RequestException as error:

            print(
                f"ERROR downloading "
                f"{satellite_info['name']}"
            )

            print(error)

            print(
                "Skipping this satellite "
                "and continuing..."
            )

            continue

        except Exception as error:

            print(
                f"UNEXPECTED ERROR: "
                f"{satellite_info['name']}"
            )

            print(error)

            print(
                "Skipping this satellite "
                "and continuing..."
            )

            continue


if __name__ == "__main__":

    main()