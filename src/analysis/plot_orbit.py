import matplotlib.pyplot as plt
import numpy as np


def plot_orbit(
    sgp4_df,
    satellite_name,
    save_path=None,
    cpf_df=None
):
    """
    Plot satellite orbit.

    Parameters
    ----------
    sgp4_df : DataFrame
        SGP4 propagated positions.

    satellite_name : str
        Satellite name.

    save_path : str
        Output image path.

    cpf_df : DataFrame, optional
        CPF reference positions.
        If provided, CPF orbit is plotted in red.
    """

    fig = plt.figure(figsize=(9, 9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # -------------------------
    # Plot SGP4 orbit
    # -------------------------

    ax.plot(
        sgp4_df["x"],
        sgp4_df["y"],
        sgp4_df["z"],
        linewidth=2,
        color="blue",
        label="SGP4 Prediction",
    )


    # -------------------------
    # Plot CPF reference
    # -------------------------

    if cpf_df is not None:

        ax.plot(
            cpf_df["x"],
            cpf_df["y"],
            cpf_df["z"],
            linewidth=2,
            color="red",
            linestyle="--",
            label="CPF Reference",
        )


    # -------------------------
    # Earth
    # -------------------------

    ax.scatter(
        0,
        0,
        0,
        s=120,
        color="green",
        label="Earth",
    )


    # -------------------------
    # Equal axis scaling
    # -------------------------

    all_x = list(sgp4_df["x"])
    all_y = list(sgp4_df["y"])
    all_z = list(sgp4_df["z"])


    if cpf_df is not None:

        all_x.extend(cpf_df["x"])
        all_y.extend(cpf_df["y"])
        all_z.extend(cpf_df["z"])


    max_range = np.array(
        [
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z),
        ]
    ).max() / 2


    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_z = (max(all_z) + min(all_z)) / 2


    ax.set_xlim(
        mid_x - max_range,
        mid_x + max_range
    )

    ax.set_ylim(
        mid_y - max_range,
        mid_y + max_range
    )

    ax.set_zlim(
        mid_z - max_range,
        mid_z + max_range
    )


    # -------------------------
    # Labels
    # -------------------------

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")


    ax.set_title(
        f"{satellite_name}\nSGP4 vs CPF Orbit"
    )


    ax.legend()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )


    plt.show()