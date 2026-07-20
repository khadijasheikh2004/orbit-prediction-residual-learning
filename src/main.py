import os
import pandas as pd

from read_tle import read_tle
from read_cpf import read_cpf
from propagate import propagate_satellite
from save_predictions import save_predictions
from plot_orbit import plot_orbit


TLE_FILE = "../tle/satellites.tle"
CPF_FOLDER = "../cpf"
OUTPUT_FOLDER = "../output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


CPF_FILES = {
    "AJISAI (EGS)": "ajisai.cpf",
    "EXPLORER 27 (BEACON-C)": "beaconc.cpf",
    "COSMOS 1989 (ETALON 1)": "etalon1.cpf",
    "COSMOS 2024 (ETALON 2)": "etalon2.cpf",
    "LAGEOS 1": "lageos1.cpf",
    "LAGEOS 2": "lageos2.cpf",
    "LARES": "lares.cpf",
    "LARETS": "larets.cpf",
    "STARLETTE": "starlette.cpf",
    "STELLA": "stella.cpf",
}


def main():

    print("=" * 60)
    print("Residual Learning Physics Pipeline")
    print("=" * 60)

    satellites = read_tle(TLE_FILE)

    all_predictions = []
    all_residuals = []

    for sat in satellites:

        name = sat["name"]

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        cpf_file = CPF_FILES.get(name)

        if cpf_file is None:
            print("No CPF mapping found.")
            continue

        cpf_path = os.path.join(CPF_FOLDER, cpf_file)

        if not os.path.exists(cpf_path):
            print(f"Missing CPF file: {cpf_file}")
            continue

        cpf_df = read_cpf(cpf_path)

        print(f"CPF state vectors: {len(cpf_df)}")

        predictions = propagate_satellite(
            sat["satellite"],
            cpf_df["timestamp"]
        )

        sgp4_df = pd.DataFrame(predictions)

        sgp4_df["satellite"] = name

        merged = pd.merge(
            sgp4_df,
            cpf_df,
            on="timestamp",
            suffixes=("_sgp4", "_cpf"),
        )

        merged["dx"] = merged["x_cpf"] - merged["x_sgp4"]
        merged["dy"] = merged["y_cpf"] - merged["y_sgp4"]
        merged["dz"] = merged["z_cpf"] - merged["z_sgp4"]

        merged["residual"] = (
            merged["dx"] ** 2
            + merged["dy"] ** 2
            + merged["dz"] ** 2
        ) ** 0.5

        all_predictions.append(sgp4_df)
        all_residuals.append(merged)

        safe_name = (
            name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        plot_orbit(
            sgp4_df,
            name,
            os.path.join(
                OUTPUT_FOLDER,
                f"{safe_name}_orbit.png",
            ),
            cpf_df,
        )

        merged.to_csv(
            os.path.join(
                OUTPUT_FOLDER,
                f"{safe_name}_residuals.csv",
            ),
            index=False,
        )

    if all_predictions:

        prediction_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        save_predictions(
            prediction_df,
            os.path.join(
                OUTPUT_FOLDER,
                "sgp4_predictions.csv",
            ),
        )

    if all_residuals:

        residual_df = pd.concat(
            all_residuals,
            ignore_index=True,
        )

        residual_df.to_csv(
            os.path.join(
                OUTPUT_FOLDER,
                "all_residuals.csv",
            ),
            index=False,
        )

        print(f"\nSaved {len(residual_df)} residual records.")

    print("\nFinished successfully.")


if __name__ == "__main__":
    main()