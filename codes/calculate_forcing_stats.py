"""
Calculate forcing data statistics (mean and std) for CAMELS dataset

Usage:
    python calculate_forcing_stats.py --camels_path /path/to/camels/data --basin_list_path /path/to/basin_list.txt --train_start 1980-10-01 --train_end 1990-09-30

    Or import as module:

    from codes.calculate_forcing_stats import calculate_forcing_stats

    stats = calculate_forcing_stats(
        camels_path="autodl-tmp/data/CAMELS",
        basin_list_path="autodl-tmp/data/basin_list.txt",
        train_start = "1980-10-01",
        train_end = "1990-09-30",
        forcing_sources = ['daymet', 'nldas', 'maurer']
        )

"""

import pandas as pd
import numpy as np
import dateutil.parser
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Union
from tqdm import tqdm


def calculate_forcing_stats(
        camels_path: Union[str, Path],
        basin_list_path: Union[str, Path],
        train_start: str = "1980-10-01",
        train_end: str = "1990-09-30",
        forcing_sources: Optional[List[str]] = None
) -> Dict:
    """
    Calculate forcing data statistics for specified training period

    Args:
        camels_path: CAMELS data directory path
        basin_list_path: Basin list file path
        train_start: Training start date (format: YYYY-MM-DD), default "1980-10-01"
        train_end: Training end date (format: YYYY-MM-DD), default "1990-09-30"
        forcing_sources: List of forcing sources, default ['daymet', 'nldas', 'maurer']

    Returns:
        Nested dictionary containing mean and std values

    Raises:
        FileNotFoundError: When specified paths don't exist
        ValueError: When date format is incorrect
    """
    camels_path = Path(camels_path)
    basin_list_path = Path(basin_list_path)

    if not camels_path.exists():
        raise FileNotFoundError(f"CAMELS data directory not found: {camels_path}")
    if not basin_list_path.exists():
        raise FileNotFoundError(f"Basin list file not found: {basin_list_path}")

    if forcing_sources is None:
        forcing_sources = ['daymet', 'nldas', 'maurer']

    all_forcings = {}
    for source in forcing_sources:
        all_forcings[source] = {
            'prcp(mm/day)': [],
            'srad(W/m2)': [],
            'tmax(C)': [],
            'tmin(C)': [],
            'vp(Pa)': []
        }

    try:
        basin_list = np.loadtxt(basin_list_path, dtype=str, encoding='utf-8')
        print(f"Successfully loaded basin list: {len(basin_list)} basins")
    except Exception as e:
        raise FileNotFoundError(f"Cannot read basin list file: {e}")

    try:
        start_date = dateutil.parser.parse(train_start)
        end_date = dateutil.parser.parse(train_end)
        print(f"Training period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        raise ValueError(f"Date format error: {e}")

    for source in forcing_sources:
        print(f"\nProcessing source: {source}")
        processed_basins = 0
        failed_basins = 0

        for basin in tqdm(basin_list, desc=f"Processing {source}"):
            forcing_path = camels_path / 'basin_mean_forcing' / source

            if not forcing_path.exists():
                print(f"Warning: Source directory not found {forcing_path}")
                break

            files = list(forcing_path.glob('**/*_forcing_leap.txt'))
            file_path = [f for f in files if f.name.startswith(basin + '_')]

            if len(file_path) == 0:
                print(f'No file for Basin {basin} in source {source}')
                failed_basins += 1
                continue
            else:
                file_path = file_path[0]

            try:
                col_names = ['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)',
                             'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']

                df = pd.read_csv(file_path, sep=r'\s+', header=3, names=col_names)

                dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
                df.index = pd.to_datetime(dates, format="%Y/%m/%d")

                df = df[start_date:end_date]

                if len(df) == 0:
                    print(f'Warning: No data for Basin {basin} in specified date range')
                    failed_basins += 1
                    continue

                for key in all_forcings[source].keys():
                    if key in df.columns:
                        values = df[key].dropna().values
                        all_forcings[source][key].extend(values)

                processed_basins += 1

            except Exception as e:
                print(f'Error processing Basin {basin} (source: {source}): {str(e)}')
                failed_basins += 1
                continue

        print(f"Source {source}: processed {processed_basins} basins, failed {failed_basins} basins")

    print("\nCalculating statistics...")
    mean_std_values = {}
    for source, data in all_forcings.items():
        mean_std_values[source] = {}
        for key, values in data.items():
            if len(values) > 0:
                mean_std_values[source][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
                print(
                    f"{source} - {key}: samples={len(values)}, mean={mean_std_values[source][key]['mean']:.4f}, std={mean_std_values[source][key]['std']:.4f}")
            else:
                print(f"Warning: No data found for {source} - {key}")
                mean_std_values[source][key] = {
                    'mean': 0.0,
                    'std': 1.0
                }

    print("\n" + "=" * 60)
    print("COPY-PASTE READY DICTIONARY:")
    print("=" * 60)
    print(mean_std_values)
    print("=" * 60)

    return mean_std_values


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Calculate forcing data statistics for CAMELS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python calculate_forcing_stats.py --camels_path /path/to/camels/data --basin_list_path /path/to/basin_list.txt --train_start 1980-10-01 --train_end 1990-09-30
        """
    )

    parser.add_argument(
        '--camels_path',
        type=str,
        required=True,
        help='CAMELS data directory path'
    )

    parser.add_argument(
        '--basin_list_path',
        type=str,
        required=True,
        help='Basin list file path'
    )

    parser.add_argument(
        '--train_start',
        type=str,
        default="1980-10-01",
        help='Training start date (format: YYYY-MM-DD), default: 1980-10-01'
    )

    parser.add_argument(
        '--train_end',
        type=str,
        default="1990-09-30",
        help='Training end date (format: YYYY-MM-DD), default: 1990-09-30'
    )

    parser.add_argument(
        '--forcing_sources',
        nargs='+',
        default=['daymet', 'nldas', 'maurer'],
        help='Forcing source list (default: daymet nldas maurer)'
    )

    args = parser.parse_args()

    try:
        results = calculate_forcing_stats(
            camels_path=args.camels_path,
            basin_list_path=args.basin_list_path,
            train_start=args.train_start,
            train_end=args.train_end,
            forcing_sources=args.forcing_sources
        )

        print("\n" + "=" * 50)
        print("Calculation complete!")
        print("=" * 50)

        for source in results:
            print(f"\n{source} source statistics:")
            for var, stats in results[source].items():
                print(f"  {var}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
