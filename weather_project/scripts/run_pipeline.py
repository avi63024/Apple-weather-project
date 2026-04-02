from __future__ import annotations

from analysis import run_analysis
from db_loader import load_weather_data


def main() -> None:
    print("Step 1/2: Loading weather data into SQLite...")
    load_weather_data()
    print()
    print("Step 2/2: Running analysis and generating charts...")
    run_analysis()
    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
