import pandas as pd
import numpy as np
import dateutil.parser
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def load_forcing(
        camels_path: Path,
        basin_list: List[str],
        is_train: bool = True,
        is_valid: bool = False,
        date_ranges: Optional[Dict[str, str]] = None,
        forcing_sources=None,
        mean_std_values: Optional[Dict[str, Dict[str, Dict[str, float]]]]=None
) -> Tuple[np.ndarray, int]:
    """
    Loads forcing data from different sources, standardizes it, and returns the data and basin area.

    :param camels_path: Path to the CAMELS dataset.
    :param basin_list: List of basin names (multiple basins supported).
    :param is_train: Whether to load training data, default is True.
    :param is_valid: Whether to load validation data, default is False.
    :param date_ranges: Dictionary containing the start and end dates for train, validation, and test.
    :param forcing_sources: Dictionary where keys are data source names, and values are lists of variables to load for each source.
    :param mean_std_values: Predefined dictionary containing mean and std values for normalization.
    :return: Standardized forcing data (2D array) and the basin area.
    """
    if forcing_sources is None:
        forcing_sources = {'daymet': ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                           'nldas':  ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                           'maurer': ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
                           }
    combined_data = []
    basin = ''.join(basin_list)
    area = None

    if mean_std_values is None:
        raise ValueError("mean_std_values parameter is required for data standardization")

    for source, variables in forcing_sources.items():
        forcing_path = camels_path / 'basin_mean_forcing' / source
        files = list(forcing_path.glob('**/*_forcing_leap.txt'))
        file_path = [f for f in files if f.name[:8] == basin]

        if len(file_path) == 0:
            raise RuntimeError(f'No file for Basin {basin_list} in source {source}')
        else:
            file_path = file_path[0]

        with open(file_path, 'r') as fp:
            fp.readline()
            fp.readline()
            area = int(fp.readline().strip())

        col_names = ['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)',
                     'tmin(C)', 'vp(Pa)']
        df = pd.read_csv(file_path, sep=r'\s+', header=3, names=col_names)
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        if is_train:
            if date_ranges:
                df = df[
                     dateutil.parser.parse(date_ranges["train_start"]):dateutil.parser.parse(date_ranges["train_end"])]
        elif is_valid:
            if date_ranges:
                df = df[
                     dateutil.parser.parse(date_ranges["valid_start"]):dateutil.parser.parse(date_ranges["valid_end"])]
        else:
            if date_ranges:
                df = df[dateutil.parser.parse(date_ranges["test_start"]):dateutil.parser.parse(date_ranges["test_end"])]

        if source not in mean_std_values:
            raise ValueError(f"No statistics found for source {source}")

        mean_std = mean_std_values[source]

        standardized_data = []
        for var in variables:
            if var not in mean_std:
                raise ValueError(f"No statistics found for variable {var} in source {source}")

            standardized_var = (df[var].values - mean_std[var]['mean']) / mean_std[var]['std']
            standardized_data.append(standardized_var)

        X = np.array(standardized_data).T.astype(np.float32)
        combined_data.append(X)

    combined_data = np.concatenate(combined_data, axis=1)

    return combined_data, area
