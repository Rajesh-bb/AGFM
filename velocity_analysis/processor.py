
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def process_velocity_rasters(config):
    """
    Process all 3D velocity rasters using paths from pipeline_config.json.
    """

    # Read paths from the existing AGFM configuration
    output_dir = Path(config["save_path"])
    velocity_dir = output_dir / "3D_Velocities"
    shapefile_path = Path(config["shapefile_dir"])

    if not velocity_dir.is_dir():
        raise FileNotFoundError(f"Velocity folder not found: {velocity_dir}")

    if not shapefile_path.is_file():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    # Read glacier boundary
    gdf = gpd.read_file(shapefile_path)

    if gdf.empty or gdf.crs is None:
        raise ValueError("Invalid shapefile or missing CRS.")

    results = []

    # Read all velocity TIFF files
    for tif in sorted(velocity_dir.glob("*.tif")):

        if tif.stem.endswith("_unc"):
            continue

        parts = tif.stem.split("_")
        start_date = datetime.strptime(parts[0], "%Y%m%d").date()
        end_date = datetime.strptime(parts[1], "%Y%m%d").date()

        with rasterio.open(tif) as src:

            if src.count < 3:
                raise ValueError(f"Expected 3 bands: {tif.name}")

            if src.crs is None:
                raise ValueError(f"Missing CRS: {tif.name}")

            shapes = gdf.to_crs(src.crs).geometry
            shapes = [
                geom for geom in shapes
                if geom is not None and not geom.is_empty
            ]

            data, _ = mask(
                src, shapes, crop=True, filled=False
            )

            N, E, U = [
                np.ma.asarray(data[i]).filled(np.nan).astype(float)
                for i in range(3)
            ]

            valid = (
                np.isfinite(N) &
                np.isfinite(E) &
                np.isfinite(U)
            )

            if not valid.any():
                print(f"Skipping: {tif.name}")
                continue

            horizontal = np.hypot(N, E)
            speed_3d = np.sqrt(N**2 + E**2 + U**2)

            h = horizontal[valid]
            v3d = speed_3d[valid]

            results.append({
                "Start_Date": start_date,
                "End_Date": end_date,
                "Valid_Pixels": int(valid.sum()),
                "Mean_North_m_per_year": np.mean(N[valid]),
                "Mean_East_m_per_year": np.mean(E[valid]),
                "Mean_Vertical_m_per_year": np.mean(U[valid]),
                "Mean_Horizontal_m_per_year": np.mean(h),
                "Median_Horizontal_m_per_year": np.median(h),
                "Std_Horizontal_m_per_year": np.std(h),
                "P90_Horizontal_m_per_year": np.percentile(h, 90),
                "P95_Horizontal_m_per_year": np.percentile(h, 95),
                "Max_Horizontal_m_per_year": np.max(h),
                "Mean_3D_m_per_year": np.mean(v3d),
                "Median_3D_m_per_year": np.median(v3d)
            })

        print(f"Processed: {tif.name}")

    if not results:
        raise RuntimeError("No valid velocity rasters processed.")

    return pd.DataFrame(results).sort_values("Start_Date").reset_index(drop=True)
