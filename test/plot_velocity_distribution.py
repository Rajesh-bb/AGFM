
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# AGFM repository root
ROOT = Path(__file__).resolve().parent.parent

# Load existing configuration
with open(ROOT / "configs" / "pipeline_config.json") as f:
    config = json.load(f)

velocity_dir = Path(config["save_path"]) / "3D_Velocities"
shapefile = config["shapefile_dir"]

plot_dir = ROOT / "test" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# Select the first velocity raster
rasters = sorted(
    p for p in velocity_dir.glob("*.tif")
    if not p.stem.endswith("_unc")
)

if not rasters:
    raise FileNotFoundError("No velocity TIFF files found.")

tif = rasters[0]

print("Reading:", tif.name)

# Read glacier boundary
gdf = gpd.read_file(shapefile)

if gdf.empty or gdf.crs is None:
    raise ValueError("Invalid shapefile or missing CRS.")


# Read velocity raster and apply glacier mask
with rasterio.open(tif) as src:

    shapes = gdf.to_crs(src.crs).geometry
    shapes = [
        geom for geom in shapes
        if geom is not None and not geom.is_empty
    ]

    data, _ = mask(
        src,
        shapes,
        crop=True,
        filled=False
    )

    # Band 1 = North, Band 2 = East
    N = np.ma.asarray(data[0], dtype=np.float64)
    E = np.ma.asarray(data[1], dtype=np.float64)

    # Calculate horizontal velocity
    speed = np.ma.sqrt(N**2 + E**2)

    # Extract valid pixel values
    values = speed.compressed()
    values = values[np.isfinite(values)]

if values.size == 0:
    raise RuntimeError("No valid velocity pixels found.")


# Calculate distribution statistics
mean = np.mean(values)
median = np.median(values)
p95 = np.percentile(values, 95)

print("\nVelocity statistics:")
print(f"Valid pixels : {len(values)}")
print(f"Mean         : {mean:.2f} m/year")
print(f"Median       : {median:.2f} m/year")
print(f"95th percentile: {p95:.2f} m/year")
print(f"Maximum      : {np.max(values):.2f} m/year")


# Plot 1: Full velocity distribution
plt.figure(figsize=(10, 6))
plt.hist(values, bins=60, edgecolor="black", alpha=0.8)

plt.axvline(mean, color="red", linestyle="--",
            label=f"Mean = {mean:.2f}")
plt.axvline(median, color="green", linestyle="--",
            label=f"Median = {median:.2f}")

plt.xlabel("Horizontal velocity (m/year)")
plt.ylabel("Number of pixels")
plt.title(f"Velocity Distribution - {tif.stem}")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

full_path = plot_dir / "velocity_distribution_full.png"
plt.savefig(full_path, dpi=200)
plt.close()


# Plot 2: Zoomed distribution (up to 95th percentile)
zoom_values = values[values <= p95]

plt.figure(figsize=(10, 6))
plt.hist(zoom_values, bins=50, edgecolor="black", alpha=0.8)

plt.axvline(median, color="green", linestyle="--",
            label=f"Median = {median:.2f}")
plt.axvline(mean, color="red", linestyle="--",
            label=f"Mean = {mean:.2f}")

plt.xlabel("Horizontal velocity (m/year)")
plt.ylabel("Number of pixels")
plt.title(f"Velocity Distribution - Central 95% ({tif.stem})")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

zoom_path = plot_dir / "velocity_distribution_zoomed.png"
plt.savefig(zoom_path, dpi=200)
plt.close()


print("\nPlots saved successfully:")
print(full_path)
print(zoom_path)
