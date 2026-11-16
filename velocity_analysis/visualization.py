
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def generate_velocity_outputs(df, config):
    """
    Generate velocity CSV files and graphs.

    Inputs:
        df     : DataFrame returned by processor.py
        config : AGFM pipeline configuration dictionary
    """

    # Create output directory using AGFM configuration
    output_dir = Path(config["save_path"]) / "Velocity_Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("No velocity data available for visualization.")

    df = df.copy()
    df["Start_Date"] = pd.to_datetime(df["Start_Date"])

    # 1. Save interval-wise velocity statistics
    timeseries_csv = output_dir / "velocity_timeseries.csv"
    df.to_csv(timeseries_csv, index=False)

    # 2. Calculate yearly average velocity
    df["Year"] = df["Start_Date"].dt.year

    yearly = df.groupby("Year")[
        [
            "Mean_Horizontal_m_per_year",
            "Median_Horizontal_m_per_year",
            "Mean_3D_m_per_year"
        ]
    ].mean().reset_index()

    yearly_csv = output_dir / "yearly_summary.csv"
    yearly.to_csv(yearly_csv, index=False)

    # 3. Plot horizontal velocity time series
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Start_Date"],
        df["Mean_Horizontal_m_per_year"],
        label="Mean Horizontal Velocity"
    )

    plt.plot(
        df["Start_Date"],
        df["Median_Horizontal_m_per_year"],
        label="Median Horizontal Velocity"
    )

    plt.xlabel("Date")
    plt.ylabel("Velocity (m/year)")
    plt.title("South Lhonak Glacier - Horizontal Velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    horizontal_graph = output_dir / "horizontal_velocity_timeseries.png"
    plt.savefig(horizontal_graph, dpi=200)
    plt.close()

    # 4. Plot 3D velocity time series
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Start_Date"],
        df["Mean_3D_m_per_year"],
        label="Mean 3D Velocity"
    )

    plt.xlabel("Date")
    plt.ylabel("Velocity (m/year)")
    plt.title("South Lhonak Glacier - 3D Velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_3d = output_dir / "3d_velocity_timeseries.png"
    plt.savefig(graph_3d, dpi=200)
    plt.close()

    print("\nVelocity analysis outputs saved:")
    print(f"Time series CSV : {timeseries_csv}")
    print(f"Yearly CSV      : {yearly_csv}")
    print(f"Horizontal graph: {horizontal_graph}")
    print(f"3D graph        : {graph_3d}")

    return yearly
