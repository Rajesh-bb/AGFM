
import json
import argparse
from pathlib import Path

from .processor import process_velocity_rasters
from .visualization import generate_velocity_outputs


def main():

    # Find AGFM repository root
    repo_root = Path(__file__).resolve().parent.parent

    # Read configuration file path
    parser = argparse.ArgumentParser(
        description="AGFM Glacier Velocity Analysis"
    )

    parser.add_argument(
        "--config",
        default=str(repo_root / "configs" / "pipeline_config.json"),
        help="Path to AGFM pipeline configuration"
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    # Load AGFM configuration
    with open(config_path, "r") as file:
        config = json.load(file)

    print("\nStarting Glacier Velocity Analysis...")
    print(f"Configuration: {config_path}")

    # Step 1: Process velocity rasters
    df = process_velocity_rasters(config)

    print(f"\nProcessed {len(df)} velocity intervals.")

    # Step 2: Generate CSV files and graphs
    yearly_summary = generate_velocity_outputs(df, config)

    print("\nYearly velocity summary:")
    print(yearly_summary.to_string(index=False))

    print("\nGlacier Velocity Analysis completed successfully!")


if __name__ == "__main__":
    main()
